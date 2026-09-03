"""Apply Phase I decisions at the vLLM scheduler boundary."""

from __future__ import annotations

import contextlib
import time
from typing import Any

from waveslice.vllm.common import (
    compute_long_prefill_threshold as _compute_long_prefill_threshold,
    rebuild_queue_like as _rebuild_queue_like,
    reorder_queue as _reorder_queue,
    restore_hidden_queue_items as _restore_hidden_queue_items,
    safe_request_id as _safe_request_id,
)
from waveslice.vllm.phase1_cohorts import _phase1_collect_secondary_lora_caps
from waveslice.vllm.phase1_math import (
    compute_budget as _compute_budget,
    compute_explicit_plan_budget as _compute_explicit_plan_budget,
    phase1_effective_short_token_mass as _phase1_effective_short_token_mass,
)
from waveslice.vllm.phase1_planning import _phase1_schedule_plan
from waveslice.vllm.phase2_runtime import _phase12_apply_scheduler_cashout_to_queues
from waveslice.vllm.request_hooks import (
    _install_v1_waiting_cap_runtime_hooks,
    _restore_v1_waiting_cap_runtime_hooks,
)
from waveslice.vllm.state import Phase1ScheduleDecision, RuntimeState


def _phase1_prune_ingress_virtual_caps(state: RuntimeState) -> dict[str, int]:
    active_request_ids = {
        str(req_id)
        for (req_id, prompt_tokens) in getattr(state, "phase1_active_prompt_tokens", {}).items()
        if str(req_id) and int(prompt_tokens or 0) > 0
    }
    ingress_request_ids = {
        str(req_id) for req_id in getattr(state, "phase1_ingress_virtuals", {}) if str(req_id)
    }
    keep_ids = active_request_ids | ingress_request_ids
    return {
        str(req_id): int(chunk)
        for (req_id, chunk) in state.phase1_virtual_token_caps.items()
        if str(req_id) in keep_ids and int(chunk) > 0
    }


def _phase1_rewrite_scheduler_outputs(
    *, outputs: Any, request_id: str | None, target_chunk: int
) -> tuple[Any, bool, int, int, int, int]:
    scheduled = getattr(outputs, "scheduled_seq_groups", None)
    if not isinstance(scheduled, list) or not request_id or target_chunk <= 0:
        return (outputs, False, 0, 0, 0, 0)
    changed: list[tuple[int, int]] = []
    for item in scheduled:
        group = getattr(item, "seq_group", None)
        if group is None or _safe_request_id(group) != request_id or (not group.is_prefill()):
            continue
        old = int(getattr(item, "token_chunk_size", 0) or 0)
        new = min(old, int(target_chunk))
        if old > 0 and 0 < new < old:
            item.token_chunk_size = new
            changed.append((old, new))
    delta = sum(old - new for (old, new) in changed)
    if delta:
        outputs.num_batched_tokens = max(0, int(getattr(outputs, "num_batched_tokens", 0)) - delta)
    return (
        outputs,
        bool(changed),
        len(changed),
        sum(old for (old, _) in changed),
        sum(new for (_, new) in changed),
        delta,
    )


def _phase1_apply_sequence_len_shadow(
    *, state: RuntimeState, seq_group: Any, target_chunk: int
) -> bool:
    if target_chunk <= 0 or seq_group is None:
        return False
    shadowed = False
    for seq in seq_group.get_seqs():
        if seq.is_finished() or getattr(seq, "data", None) is None:
            continue
        data = seq.data
        (computed, total) = (data.get_num_computed_tokens(), data.get_len())
        state.phase1_shadow_seq_lens[id(data)] = max(
            computed + 1, min(total, computed + target_chunk)
        )
        shadowed = True
    return shadowed


def _set_phase1_cap(state: RuntimeState, request_id: str, chunk: int) -> None:
    state.phase1_virtual_token_caps[request_id] = int(chunk)
    state.phase1_public_skip_rewrite_requests.add(request_id)
    state.metrics.record_phase1_virtual_cap_probe(target_set=True)


def _restore_hidden(owner: Any, attribute: str, hidden: list[Any], *, append: bool = False) -> None:
    if not hidden or not hasattr(owner, attribute):
        return
    queue = getattr(owner, attribute)
    if append and hasattr(queue, "extend"):
        queue.extend(hidden)
    else:
        setattr(
            owner,
            attribute,
            _restore_hidden_queue_items(queue, hidden, queue_rebuilder=_rebuild_queue_like),
        )


def _phase1_install_caps(
    *,
    state: RuntimeState,
    scheduler_obj: Any,
    decision: Phase1ScheduleDecision,
    lora_enabled: bool,
    scheduler_config: Any,
    original_budget: Any,
    original_threshold: Any,
) -> tuple[list[Any], list[tuple[Any, str, Any]]]:
    state.phase1_shadow_seq_lens.clear()
    secondary_caps = {}
    if state.policy.phase1_ingress_direct_authoritative and lora_enabled:
        secondary_caps = _phase1_collect_secondary_lora_caps(
            state=state,
            snapshot=decision.snapshot,
            primary_request_id=decision.cohort.long_req_id,
            max_wait_us=decision.max_wait_us,
            queue_len=decision.queue_len,
            scheduler_cfg=scheduler_config,
            original_budget=original_budget,
            original_threshold=original_threshold,
        )
    if decision.cohort.long_req_id:
        _set_phase1_cap(state, str(decision.cohort.long_req_id), decision.best_chunk)
    for request_id, chunk in secondary_caps.items():
        current = state.phase1_virtual_token_caps.get(request_id)
        if chunk > 0 and (current is None or current > chunk):
            _set_phase1_cap(state, request_id, chunk)
    if not state.policy.phase1_ingress_direct_authoritative:
        _phase1_apply_sequence_len_shadow(
            state=state,
            seq_group=decision.long_group,
            target_chunk=decision.best_chunk,
        )
        return [], []
    return _install_v1_waiting_cap_runtime_hooks(state, scheduler_obj)


def _phase1_apply_scheduler_limits(
    *,
    state: RuntimeState,
    scheduler_obj: Any,
    scheduler_config: Any,
    decision: Phase1ScheduleDecision,
    can_threshold: bool,
    can_budget: bool,
    original_budget: Any,
    original_threshold: Any,
) -> None:
    short_len, long_len = decision.cohort.representative_short_len, decision.cohort.long_len
    if can_threshold and scheduler_config is not None:
        threshold = _compute_long_prefill_threshold(
            decision.best_chunk, original_threshold, scheduler_obj
        )
        if threshold is not None:
            scheduler_config.long_prefill_token_threshold = threshold
    if not can_budget or scheduler_config is None:
        return
    short_mass = _phase1_effective_short_token_mass(
        decision.cohort.short_lengths,
        short_len=short_len,
        best_chunk=decision.best_chunk,
        policy=state.policy,
    )
    if decision.explicit_kind is not None:
        budget = _compute_explicit_plan_budget(
            best_chunk=decision.best_chunk,
            short_len=short_len,
            short_token_mass=short_mass,
            policy=state.policy,
            original_budget=original_budget,
            baseline_chunk=decision.baseline_chunk,
        )
    else:
        budget = _compute_budget(
            decision.best_chunk,
            short_len,
            long_len,
            short_mass,
            decision.queue_len,
            state.policy,
            original_budget,
            baseline_chunk=decision.baseline_chunk,
        )
    if budget is not None:
        scheduler_config.max_num_batched_tokens = budget


@contextlib.contextmanager
def _phase1_schedule_scope(
    *,
    state: RuntimeState,
    scheduler_obj: Any,
    lora_enabled: bool,
    can_threshold: bool,
    can_budget: bool,
    scheduler_config: Any,
    original_budget: Any,
    original_threshold: Any,
):
    base_policy = state.policy
    hidden_running: list[Any] = []
    hidden_waiting: list[Any] = []
    cap_cleanup: list[Any] = []
    cap_restore: list[tuple[Any, str, Any]] = []
    state.phase1_virtual_token_caps = (
        _phase1_prune_ingress_virtual_caps(state)
        if state.policy.phase1_ingress_direct_authoritative
        else {}
    )
    try:
        decision = _phase1_schedule_plan(
            state=state,
            scheduler_obj=scheduler_obj,
            lora_enabled=lora_enabled,
            scheduler_config=scheduler_config,
            original_budget=original_budget,
            original_threshold=original_threshold,
        )
        if decision is not None:
            cap_cleanup, cap_restore = _phase1_install_caps(
                state=state,
                scheduler_obj=scheduler_obj,
                decision=decision,
                lora_enabled=lora_enabled,
                scheduler_config=scheduler_config,
                original_budget=original_budget,
                original_threshold=original_threshold,
            )
            _phase1_apply_scheduler_limits(
                state=state,
                scheduler_obj=scheduler_obj,
                scheduler_config=scheduler_config,
                decision=decision,
                can_threshold=can_threshold,
                can_budget=can_budget,
                original_budget=original_budget,
                original_threshold=original_threshold,
            )
            scheduler_obj.running, scheduler_obj.waiting, hidden_running, hidden_waiting, _ = (
                _phase12_apply_scheduler_cashout_to_queues(
                    state=state,
                    running=scheduler_obj.running,
                    waiting=scheduler_obj.waiting,
                )
            )
        yield decision
    finally:
        _restore_v1_waiting_cap_runtime_hooks(cap_cleanup, cap_restore)
        state.phase1_shadow_seq_lens.clear()
        state.phase1_virtual_token_caps.clear()
        state.phase1_public_skip_rewrite_requests.clear()
        state.policy = base_policy
        if scheduler_config is not None:
            if isinstance(original_budget, int) and original_budget > 0:
                scheduler_config.max_num_batched_tokens = original_budget
            if isinstance(original_threshold, int) and original_threshold >= 0:
                scheduler_config.long_prefill_token_threshold = original_threshold
        _restore_hidden(scheduler_obj, "running", hidden_running)
        _restore_hidden(scheduler_obj, "waiting", hidden_waiting)


def _phase1_reorder_scheduler_queues(state: RuntimeState, scheduler_obj: Any) -> None:
    if not state.policy.enable_sjf_reorder:
        return
    now = time.time()
    for name in ("running", "waiting"):
        setattr(
            scheduler_obj,
            name,
            _reorder_queue(
                getattr(scheduler_obj, name),
                brain=state.brain,
                now_s=now,
                mode=state.policy.queue_reorder_mode,
                aging_quantum_us=state.policy.queue_reorder_aging_quantum_us,
            ),
        )
