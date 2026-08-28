"""WaveSlice runtime integration for vLLM V1.

This module implements:
- Phase I: scheduler-side chunk-size control that returns to the
  scheduling boundary earlier for long prefills.
- Phase II: scheduler-bound priority cashout that reshapes the next scheduled
  window without delaying already-computed outputs.
- Runtime metrics hooks: TTFT / P99 / slowdown accounting.

Hook installation and restoration are managed by ``waveslice.vllm.integration``.
"""

from __future__ import annotations

import contextlib
import functools
import time
from collections.abc import Callable, Iterable
from typing import Any

from waveslice.policy import WaveSlicePolicy
from waveslice.vllm.common import (
    collect_live_snapshot as _collect_live_snapshot,
    compute_long_prefill_threshold as _compute_long_prefill_threshold,
    estimate_solo_us as _estimate_solo_us,
    has_running_waiting_queues as _has_running_waiting_queues,
    infer_lora_rank as _infer_lora_rank,
    phase2_scheduler_cashout_enabled as _phase2_scheduler_cashout_enabled,
    phase12_expected_chunk_tokens as _phase12_expected_chunk_tokens,
    rebuild_queue_like as _rebuild_queue_like,
    reorder_queue as _reorder_queue,
    restore_hidden_queue_items as _restore_hidden_queue_items,
    safe_first_seq as _safe_first_seq,
    safe_lora_path as _safe_lora_path,
    safe_prefill_uncomputed_tokens as _safe_prefill_uncomputed_tokens,
    safe_remaining_tokens as _safe_remaining_tokens,
    safe_request_id as _safe_request_id,
    safe_total_tokens as _safe_total_tokens,
)
from waveslice.vllm.phase1_math import (
    compute_budget as _compute_budget,
    compute_explicit_plan_budget as _compute_explicit_plan_budget,
    maybe_force_phase1_chunk as _maybe_force_phase1_chunk,
    need_wave_slice as _need_wave_slice,
    phase1_adjusted_queue_len as _phase1_adjusted_queue_len,
    phase1_authoritative_chunk as _phase1_authoritative_chunk,
    phase1_authoritative_short_floor as _phase1_authoritative_short_floor,
    phase1_baseline_chunk_proxy as _phase1_baseline_chunk_proxy,
    phase1_cohort_target_len as _phase1_cohort_target_len,
    phase1_effective_ingress_min_chunk as _phase1_effective_ingress_min_chunk,
    phase1_effective_short_token_mass as _phase1_effective_short_token_mass,
    phase1_runtime_adapt_policy as _phase1_runtime_adapt_policy,
    phase1_runtime_pressure_meta as _phase1_runtime_pressure_meta,
)
from waveslice.vllm.phase1_selection import (
    phase1_find_ingress_virtual_candidate as _phase1_find_ingress_virtual_candidate,
    phase1_find_seq_group_by_request_id as _phase1_find_seq_group_by_request_id,
    phase1_live_cohort_from_snapshot as _phase1_live_cohort_from_snapshot,
)
from waveslice.vllm.phase1_state import (
    phase1_direct_chunk_candidate as _phase1_direct_chunk_candidate,
    phase1_explicit_chunk_from_plan as _phase1_explicit_chunk_from_plan,
)
from waveslice.vllm.phase2_beneficiaries import (
    phase2_beneficiary_signal as _phase2_beneficiary_signal,
    phase12_beneficiary_signal as _phase12_beneficiary_signal,
    phase12_collect_prefill_lora_state as _phase12_collect_prefill_lora_state,
    phase12_collect_snapshot_req_infos as _phase12_collect_snapshot_req_infos,
)
from waveslice.vllm.phase2_cashout import (
    phase12_cashout_candidate_id as _phase12_cashout_candidate_id,
    phase12_scheduler_cashout_cooldown_for_grade as _phase12_scheduler_cashout_cooldown_for_grade,
    phase12_scheduler_cashout_grade as _phase12_scheduler_cashout_grade,
    phase12_scheduler_cashout_value_signal as _phase12_scheduler_cashout_value_signal,
)
from waveslice.vllm.phase2_gates import (
    phase2_selective_gate as _phase2_selective_gate,
    phase12_joint_phase2_ready as _phase12_joint_phase2_gate,
)
from waveslice.vllm.phase2_priority import (
    phase12_priority_bubble_waiting_queue as _phase12_priority_bubble_waiting_queue,
)
from waveslice.vllm.request_hooks import (
    _install_v1_waiting_cap_runtime_hooks,
    _restore_v1_waiting_cap_runtime_hooks,
)
from waveslice.vllm.state import (
    Phase1CohortStats,
    Phase1IngressVirtualSlice,
    Phase1ScheduleDecision,
    Phase12BeneficiarySignal,
    RuntimeState,
    ScheduledRequestInfo,
)


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


def _phase1_lora_cohort_key(seq_group: Any) -> str | None:
    lora_request = getattr(seq_group, "lora_request", None)
    if lora_request is None:
        seq = _safe_first_seq(seq_group)
        lora_request = getattr(seq, "lora_request", None) if seq is not None else None
    path = _safe_lora_path(lora_request)
    if path:
        return f"path:{path}"
    rank = int(_infer_lora_rank(lora_request) or 0)
    if rank > 0:
        return f"rank:{rank}"
    return None


def _phase1_filter_snapshot_for_lora_cohort(
    snapshot: list[tuple[Any, int]], *, preferred_request_id: str | None = None
) -> list[tuple[Any, int]]:
    if len(snapshot) < 2:
        return snapshot
    anchor = next(
        (
            pair
            for pair in snapshot
            if preferred_request_id and _safe_request_id(pair[0]) == preferred_request_id
        ),
        max(snapshot, key=lambda pair: int(pair[1])),
    )
    key = _phase1_lora_cohort_key(anchor[0])
    if not key:
        return snapshot
    filtered = [
        (group, int(remaining))
        for (group, remaining) in snapshot
        if _phase1_lora_cohort_key(group) == key
    ]
    return filtered if len(filtered) >= 2 else snapshot


def _phase1_waiting_short_count(waiting: Iterable[Any], *, short_threshold_tokens: int) -> int:
    threshold = max(1, int(short_threshold_tokens))
    return sum(
        1 for group in waiting if 0 < int(_safe_prefill_uncomputed_tokens(group) or 0) <= threshold
    )


def _phase1_build_ingress_fallback_cohort(
    state: RuntimeState, snapshot: list[tuple[Any, int]]
) -> Phase1CohortStats | None:
    if not snapshot or not state.phase1_ingress_virtuals:
        return None
    (group, remaining) = max(snapshot, key=lambda pair: int(pair[1]))
    request_id = _safe_request_id(group)
    virtual = state.phase1_ingress_virtuals.get(str(request_id))
    if virtual is None:
        return None
    short_lengths = [int(value) for value in virtual.short_lengths if int(value) > 0]
    representative = max(1, int(virtual.representative_short_len))
    short_lengths = short_lengths or [representative]
    if not _need_wave_slice(short_lengths + [int(virtual.original_long_len)], state.policy):
        return None
    return Phase1CohortStats(
        representative_short_len=representative,
        short_count=max(1, int(virtual.short_count)),
        short_token_mass=max(representative, int(virtual.short_token_mass)),
        short_lengths=short_lengths,
        long_len=max(1, int(remaining)),
        long_req_id=str(request_id),
        total_count=max(2, len(snapshot), int(virtual.active_count)),
    )


def _phase1_build_global_activity_cohort(
    state: RuntimeState, snapshot: list[tuple[Any, int]]
) -> Phase1CohortStats | None:
    live: dict[str, int] = {}
    for group, remaining in snapshot:
        request_id = _safe_request_id(group)
        if request_id and int(remaining) > live.get(str(request_id), 0):
            live[str(request_id)] = int(remaining)
    active = {
        str(request_id): int(tokens)
        for (request_id, tokens) in state.phase1_active_prompt_tokens.items()
        if request_id and int(tokens) > 0
    }
    virtuals = state.phase1_ingress_virtuals
    if not live or not (active or virtuals):
        return None
    best = None
    best_score = None
    for request_id, remaining in live.items():
        virtual = virtuals.get(request_id)
        original_long = max(
            remaining, int(getattr(virtual, "original_long_len", active.get(request_id, remaining)))
        )
        short_lengths = [
            int(value) for value in getattr(virtual, "short_lengths", []) if int(value) > 0
        ] or sorted(tokens for (rid, tokens) in active.items() if rid != request_id)
        if not short_lengths or not _need_wave_slice(short_lengths + [original_long], state.policy):
            continue
        selected = [min(short_lengths)]
        representative = max(1, round(sum(selected) / len(selected)))
        candidate = Phase1CohortStats(
            representative_short_len=representative,
            short_count=max(1, int(getattr(virtual, "short_count", len(selected)))),
            short_token_mass=max(
                representative, int(getattr(virtual, "short_token_mass", sum(selected)))
            ),
            short_lengths=list(selected),
            long_len=remaining,
            long_req_id=request_id,
            total_count=max(
                len(active),
                len(selected) + 1,
                int(getattr(virtual, "active_count", len(active) or len(selected) + 1)),
            ),
        )
        score = (remaining, original_long, candidate.total_count)
        if best_score is None or score > best_score:
            (best, best_score) = (candidate, score)
    return best


def _phase1_group_snapshot_by_lora_cohort(
    snapshot: list[tuple[Any, int]],
) -> dict[str, list[tuple[Any, int]]]:
    grouped: dict[str, list[tuple[Any, int]]] = {}
    for seq_group, rem in snapshot:
        key = _phase1_lora_cohort_key(seq_group)
        if not key:
            continue
        grouped.setdefault(key, []).append((seq_group, int(rem)))
    return grouped


def _phase1_candidate_chunk_for_snapshot(
    *,
    state: RuntimeState,
    snapshot: list[tuple[Any, int]],
    max_wait_us: float,
    queue_len: int,
    scheduler_cfg: Any,
    original_budget: Any,
    original_threshold: Any,
) -> tuple[Phase1CohortStats, int] | None:
    if not _need_wave_slice([remaining for (_, remaining) in snapshot], state.policy):
        return None
    cohort = _phase1_live_cohort_from_snapshot(snapshot, request_id_getter=_safe_request_id)
    if cohort is None or not cohort.long_req_id:
        return None
    long_group = _phase1_find_seq_group_by_request_id(
        snapshot, cohort.long_req_id, request_id_getter=_safe_request_id
    )
    (short_len, long_len) = (cohort.representative_short_len, cohort.long_len)
    if long_group is None or long_len <= short_len:
        return None
    queue_len = _phase1_adjusted_queue_len(cohort, queue_len, state.policy)
    baseline = _phase1_baseline_chunk_proxy(
        long_len=long_len,
        original_budget=original_budget,
        original_threshold=original_threshold,
        scheduler_cfg=scheduler_cfg,
        policy=state.policy,
    )
    best = state.slicer.choose_dynamic_chunk(
        short_len=short_len,
        long_len=long_len,
        scheduler=state.brain,
        t_wait_us=max_wait_us,
        queue_length=queue_len,
        baseline_chunk=baseline,
    )
    explicit = _phase1_explicit_chunk_from_plan(
        state=state,
        cohort=cohort,
        snapshot=snapshot,
        t_wait_us=max_wait_us,
        queue_length=queue_len,
        baseline_chunk=baseline,
        total_tokens_getter=_safe_total_tokens,
        request_id_getter=_safe_request_id,
    )
    if explicit is not None:
        best = int(explicit[0])
    if state.policy.phase1_ingress_direct_authoritative:
        total = max(1, int(_safe_total_tokens(long_group) or long_len))
        direct = _phase1_direct_chunk_candidate(
            state=state,
            cohort=cohort,
            total_len=total,
            done_offset=max(0, total - long_len),
            remaining_len=long_len,
            baseline_chunk=baseline,
        )
        if direct is not None:
            best = min(best, direct)
    if _phase1_lora_cohort_key(long_group):
        best = min(best, _phase1_cohort_target_len(cohort, state.policy))
    best = max(short_len, min(int(best), long_len - 1))
    return (cohort, best) if best < long_len else None


def _phase1_collect_secondary_lora_caps(
    *,
    state: RuntimeState,
    snapshot: list[tuple[Any, int]],
    primary_request_id: str | None,
    max_wait_us: float,
    queue_len: int,
    scheduler_cfg: Any,
    original_budget: Any,
    original_threshold: Any,
) -> dict[str, int]:
    if not snapshot:
        return {}
    caps: dict[str, int] = {}
    grouped = _phase1_group_snapshot_by_lora_cohort(snapshot)
    for cohort_snapshot in grouped.values():
        candidate = _phase1_candidate_chunk_for_snapshot(
            state=state,
            snapshot=cohort_snapshot,
            max_wait_us=max_wait_us,
            queue_len=queue_len,
            scheduler_cfg=scheduler_cfg,
            original_budget=original_budget,
            original_threshold=original_threshold,
        )
        if candidate is None:
            continue
        (cohort, best_chunk) = candidate
        req_id = str(cohort.long_req_id or "")
        if not req_id or req_id == str(primary_request_id or ""):
            continue
        caps[req_id] = min(int(best_chunk), int(caps.get(req_id, best_chunk)))
    return caps


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


def _phase12_joint_phase1_floor(
    *, state: RuntimeState, snapshot: Any, policy: WaveSlicePolicy
) -> int | None:
    if not (policy.enable_phase1_scheduler and policy.enable_phase2_scheduler):
        return None
    if not bool(policy.phase12_joint_coordination):
        return None
    if not policy.phase2_enable_scheduler_cashout:
        return None
    if hasattr(snapshot, "running") and hasattr(snapshot, "waiting"):
        seq_groups = list(snapshot.running) + list(snapshot.waiting)
    else:
        seq_groups = [seq_group for (seq_group, _) in list(snapshot or [])]
    (prefill_lens, lora_ranks) = _phase12_collect_prefill_lora_state(
        seq_groups, rank_infer=_infer_lora_rank, remaining_getter=_safe_prefill_uncomputed_tokens
    )
    if len(prefill_lens) < max(2, int(policy.phase2_min_prefill_count)):
        return None
    (selective_ok, _ratio, pressure_ratio, lora_rank_hetero) = _phase2_selective_gate(
        prefill_lens=prefill_lens, lora_ranks=lora_ranks, policy=policy
    )
    if not selective_ok:
        return None
    if not lora_rank_hetero and pressure_ratio < float(policy.phase2_min_pressure_ratio):
        return None
    return max(1, int(policy.phase12_joint_min_chunk))


def _phase12_joint_phase2_ready(
    *,
    state: RuntimeState,
    policy: WaveSlicePolicy,
    prefill_lens: list[int],
    num_decode_tokens: int,
    lora_ranks: list[int],
    req_infos: list[ScheduledRequestInfo] | None,
) -> tuple[bool, str]:
    signal = _phase12_beneficiary_signal(state=state, policy=policy, req_infos=req_infos or [])
    return _phase12_joint_phase2_gate(
        state=state,
        policy=policy,
        prefill_lens=prefill_lens,
        num_decode_tokens=num_decode_tokens,
        lora_ranks=lora_ranks,
        signal=signal,
    )


def _phase12_tick_recent_phase1(state: RuntimeState) -> None:
    recent_ttl = int(getattr(state, "phase12_recent_phase1_apply_ttl", 0) or 0)
    if recent_ttl > 0:
        state.phase12_recent_phase1_apply_ttl = max(0, recent_ttl - 1)
        state.phase12_recent_phase1_strength = max(
            0.0, float(getattr(state, "phase12_recent_phase1_strength", 0.0) or 0.0) * 0.7
        )
    else:
        state.phase12_recent_phase1_strength = 0.0
        state.phase12_recent_phase1_chunk = 0


def _phase12_tick_recent_phase2(state: RuntimeState) -> None:
    cooldown = int(getattr(state, "phase12_recent_phase2_cashout_cooldown", 0) or 0)
    if cooldown > 0:
        state.phase12_recent_phase2_cashout_cooldown = max(0, cooldown - 1)
    lane_ttl = int(getattr(state, "phase2_priority_lane_ttl", 0) or 0)
    if lane_ttl > 0:
        state.phase2_priority_lane_ttl = max(0, lane_ttl - 1)
    if int(getattr(state, "phase2_priority_lane_ttl", 0) or 0) <= 0:
        state.phase2_priority_active_ids.clear()
        state.phase2_priority_deferred_ids.clear()


def _phase12_activate_priority_lane(
    state: RuntimeState,
    *,
    beneficiary_ids: Iterable[str],
    deferred_ids: Iterable[str],
    lane_ttl: int | None = None,
) -> None:
    active = {str(rid) for rid in beneficiary_ids if str(rid)}
    deferred = {str(rid) for rid in deferred_ids if str(rid)}
    state.phase2_priority_active_ids = active
    state.phase2_priority_deferred_ids = deferred
    base_ttl = (
        int(lane_ttl)
        if lane_ttl is not None
        else int(state.policy.phase12_phase2_priority_lane_ttl)
    )
    state.phase2_priority_lane_ttl = max(1, base_ttl)
    state.metrics.record_priority_lane_activation(
        active_ids=active, deferred_ids=deferred, lane_ttl=state.phase2_priority_lane_ttl
    )


def _cashout_context(
    state: RuntimeState, snapshot: list[tuple[Any, int]], *, num_decode_tokens: int
) -> tuple[list[ScheduledRequestInfo], Phase12BeneficiarySignal | None, str | None]:
    infos = _phase12_collect_snapshot_req_infos(
        snapshot,
        state=state,
        policy=state.policy,
        request_id_getter=_safe_request_id,
        expected_chunk_getter=_phase12_expected_chunk_tokens_adapter,
        rank_infer=_infer_lora_rank,
    )
    (lengths, ranks) = _phase12_collect_prefill_lora_state(
        [group for (group, _) in snapshot],
        rank_infer=_infer_lora_rank,
        remaining_getter=_safe_prefill_uncomputed_tokens,
    )
    (ready, reason) = _phase12_joint_phase2_ready(
        state=state,
        policy=state.policy,
        prefill_lens=lengths,
        num_decode_tokens=num_decode_tokens,
        lora_ranks=ranks,
        req_infos=infos,
    )
    if not ready:
        signal = None
    elif state.policy.enable_phase1_scheduler and state.policy.phase12_joint_coordination:
        signal = _phase12_beneficiary_signal(state=state, policy=state.policy, req_infos=infos)
    else:
        signal = _phase2_beneficiary_signal(policy=state.policy, req_infos=infos)
    return (infos, signal, None if ready else reason)


def _cashout_metrics(state: RuntimeState, applied: bool, reason: str) -> None:
    state.metrics.record_phase2_decision(applied, reason)


def _finish_cashout(
    state: RuntimeState,
    signal: Phase12BeneficiarySignal,
    grade: dict[str, Any],
    *,
    deferred_ids: Iterable[str] = (),
) -> None:
    _cashout_metrics(state, True, "scheduler_cashout_beneficiary")
    state.phase12_recent_phase2_cashout_cooldown = _phase12_scheduler_cashout_cooldown_for_grade(
        policy=state.policy, grade=grade
    )
    deferred = list(deferred_ids)
    if deferred:
        _phase12_activate_priority_lane(
            state, beneficiary_ids=signal.beneficiary_selected_ids[:1], deferred_ids=deferred
        )


def _phase12_scheduler_cashout_rewrite(
    *, state: RuntimeState, scheduler_outputs: Any
) -> tuple[Any, bool]:
    scheduled = getattr(scheduler_outputs, "scheduled_seq_groups", None)
    if (
        not state.policy.enable_phase2_scheduler
        or not state.policy.phase2_enable_scheduler_cashout
        or (not isinstance(scheduled, list))
        or (len(scheduled) < 2)
    ):
        return (scheduler_outputs, False)
    snapshot = [
        (item.seq_group, int(remaining))
        for item in scheduled
        if getattr(item, "seq_group", None) is not None
        and item.seq_group.is_prefill()
        and (remaining := _safe_prefill_uncomputed_tokens(item.seq_group))
        and (remaining > 1)
    ]
    if len(snapshot) < 2:
        _cashout_metrics(state, False, "scheduler_cashout_no_prefill")
        return (scheduler_outputs, False)
    (infos, signal, reason) = _cashout_context(state, snapshot, num_decode_tokens=0)
    if signal is None:
        _cashout_metrics(state, False, f"{reason}_sched")
        return (scheduler_outputs, False)
    candidate_id = _phase12_cashout_candidate_id(req_infos=infos, beneficiary_signal=signal)
    grade = _phase12_scheduler_cashout_grade(
        policy=state.policy,
        candidate_id=candidate_id,
        selected_quality=signal.beneficiary_selected_quality,
        value_signal=_phase12_scheduler_cashout_value_signal(
            req_infos=infos,
            beneficiary_signal=signal,
            candidate_id=candidate_id,
        ),
    )
    if not grade or not grade["allowed"]:
        _cashout_metrics(state, False, "scheduler_cashout_low_quality")
        return (scheduler_outputs, False)
    beneficiaries = set(signal.beneficiary_selected_ids[:1])
    (kept, removed, removed_tokens, kept_prefills) = ([], 0, 0, 0)
    for item in scheduled:
        group = getattr(item, "seq_group", None)
        if group is None or not group.is_prefill():
            kept.append(item)
            continue
        request_id = str(_safe_request_id(group) or "")
        remaining = int(_safe_prefill_uncomputed_tokens(group) or 0)
        if request_id == candidate_id and remaining > 1 and not removed:
            removed += 1
            removed_tokens += max(0, int(item.token_chunk_size or 0))
        else:
            kept.append(item)
            kept_prefills += 1
    if not removed or not beneficiaries:
        _cashout_metrics(state, False, "scheduler_cashout_not_enough_removed")
        return (scheduler_outputs, False)
    scheduler_outputs.scheduled_seq_groups = kept
    scheduler_outputs.num_batched_tokens = max(
        0, int(scheduler_outputs.num_batched_tokens) - removed_tokens
    )
    scheduler_outputs.num_prefill_groups = kept_prefills
    _finish_cashout(state, signal, grade)
    return (scheduler_outputs, True)


def _phase12_apply_scheduler_cashout_to_queues(
    *, state: RuntimeState, running: Any, waiting: Any
) -> tuple[Any, Any, list[Any], list[Any], bool]:
    empty = (running, waiting, [], [], False)
    policy = state.policy
    if not (policy.enable_phase2_scheduler and policy.phase2_enable_scheduler_cashout):
        return empty
    if state.phase12_recent_phase2_cashout_cooldown > 0:
        _cashout_metrics(state, False, "scheduler_cashout_cooldown")
        return empty
    (snapshot, _) = _collect_live_snapshot(waiting, running)
    if len(snapshot) < 2:
        _cashout_metrics(state, False, "scheduler_cashout_no_prefill")
        return empty
    combined = list(waiting) + list(running)
    decode_count = sum(
        1
        for group in combined
        if (_safe_prefill_uncomputed_tokens(group) or 0) <= 0
        and (_safe_remaining_tokens(group) or 0) > 0
    )
    (infos, signal, reason) = _cashout_context(state, snapshot, num_decode_tokens=decode_count)
    if signal is None:
        _cashout_metrics(state, False, f"{reason}_sched_pre")
        return empty
    candidate_id = _phase12_cashout_candidate_id(req_infos=infos, beneficiary_signal=signal)
    grade = _phase12_scheduler_cashout_grade(
        policy=policy,
        candidate_id=candidate_id,
        selected_quality=signal.beneficiary_selected_quality,
        value_signal=_phase12_scheduler_cashout_value_signal(
            req_infos=infos,
            beneficiary_signal=signal,
            candidate_id=candidate_id,
        ),
    )
    if not grade or not grade["allowed"]:
        _cashout_metrics(state, False, "scheduler_cashout_low_quality")
        return empty
    beneficiaries = set(signal.beneficiary_selected_ids[:1])
    hidden_running, kept_running, hidden_waiting, kept_waiting = [], [], [], []
    for groups, hidden, kept in (
        (running, hidden_running, kept_running),
        (waiting, hidden_waiting, kept_waiting),
    ):
        for group in groups:
            request_id = str(_safe_request_id(group) or "")
            should_hide = (
                not hidden_running
                and not hidden_waiting
                and request_id == candidate_id
                and request_id not in beneficiaries
                and (_safe_prefill_uncomputed_tokens(group) or 0) > 1
            )
            (hidden if should_hide else kept).append(group)
    if not hidden_running and not hidden_waiting:
        _cashout_metrics(state, False, "scheduler_cashout_not_enough_removed")
        return empty
    new_waiting = _phase12_priority_bubble_waiting_queue(
        _rebuild_queue_like(waiting, kept_waiting),
        beneficiary_signal=signal,
        beneficiary_ids=beneficiaries,
        request_id_getter=_safe_request_id,
        queue_rebuilder=_rebuild_queue_like,
    )
    deferred = [str(_safe_request_id(group)) for group in hidden_running + hidden_waiting]
    _finish_cashout(state, signal, grade, deferred_ids=deferred)
    return (
        _rebuild_queue_like(running, kept_running),
        new_waiting,
        hidden_running,
        hidden_waiting,
        True,
    )


def _phase12_expected_chunk_tokens_adapter(
    seq_group: Any, state: RuntimeState | None, remaining: int
) -> int:
    return _phase12_expected_chunk_tokens(seq_group, state=state, remaining=remaining)


def _lora_enabled(scheduler_obj: Any) -> bool:
    if getattr(scheduler_obj, "lora_config", None) is not None:
        return True
    if any(
        bool(getattr(scheduler_obj, name, False))
        for name in ("enable_lora", "lora_enabled", "has_lora", "_enable_lora")
    ):
        return True
    config = getattr(scheduler_obj, "scheduler_config", None)
    return any(bool(getattr(config, name, False)) for name in ("enable_lora", "lora_enabled"))


def _observe_scheduler_requests(state: RuntimeState, queues: Iterable[Any]) -> None:
    for seq_group in queues:
        request_id = _safe_request_id(seq_group)
        if not request_id:
            continue
        total_tokens = _safe_total_tokens(seq_group)
        state.metrics.observe_scheduler_request(
            request_id,
            total_tokens=total_tokens,
            solo_us=_estimate_solo_us(state.brain, total_tokens),
            is_short=total_tokens is not None
            and total_tokens <= state.policy.metrics_short_request_tokens,
        )


def _post_schedule_cashout(state: RuntimeState, outputs: Any) -> Any:
    if _phase2_scheduler_cashout_enabled(state.policy):
        (outputs, _) = _phase12_scheduler_cashout_rewrite(state=state, scheduler_outputs=outputs)
    return outputs


def _run_native_schedule(
    state: RuntimeState,
    scheduler_obj: Any,
    schedule_impl: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run the native scheduler while temporarily hiding Phase-II candidates."""
    hidden_running: list[Any] = []
    hidden_waiting: list[Any] = []
    if _phase2_scheduler_cashout_enabled(state.policy) and _has_running_waiting_queues(
        scheduler_obj
    ):
        (scheduler_obj.running, scheduler_obj.waiting, hidden_running, hidden_waiting, _) = (
            _phase12_apply_scheduler_cashout_to_queues(
                state=state,
                running=scheduler_obj.running,
                waiting=scheduler_obj.waiting,
            )
        )
    try:
        outputs = schedule_impl(scheduler_obj, *args, **kwargs)
    finally:
        _restore_hidden(scheduler_obj, "running", hidden_running)
        _restore_hidden(scheduler_obj, "waiting", hidden_waiting)
    return _post_schedule_cashout(state, outputs)


def _phase1_ingress_cohort(
    state: RuntimeState,
    candidate: tuple[Phase1IngressVirtualSlice, Any, int],
    snapshot: list[tuple[Any, int]],
) -> Phase1CohortStats:
    (virtual, _, remaining) = candidate
    live = _phase1_live_cohort_from_snapshot(snapshot, request_id_getter=_safe_request_id)
    short_lengths = (
        list(live.short_lengths)
        if live is not None
        else list(virtual.short_lengths) or [virtual.representative_short_len]
    )
    return Phase1CohortStats(
        representative_short_len=max(
            1,
            int(
                live.representative_short_len
                if live is not None
                else virtual.representative_short_len
            ),
        ),
        short_count=max(1, int(live.short_count if live is not None else virtual.short_count)),
        short_token_mass=max(
            1, int(live.short_token_mass if live is not None else virtual.short_token_mass)
        ),
        short_lengths=[int(value) for value in short_lengths],
        long_len=max(1, int(remaining)),
        long_req_id=str(virtual.long_req_id),
        total_count=max(
            2, int(virtual.active_count), int(live.total_count) if live is not None else 0
        ),
    )


def _select_phase1_cohort(
    state: RuntimeState, snapshot: list[tuple[Any, int]], *, lora_enabled: bool
) -> tuple[Phase1CohortStats | None, list[tuple[Any, int]], Any | None]:
    global_cohort = None if lora_enabled else _phase1_build_global_activity_cohort(state, snapshot)
    ingress = _phase1_find_ingress_virtual_candidate(
        state.phase1_ingress_virtuals, snapshot=snapshot, request_id_getter=_safe_request_id
    )
    selected = (
        _phase1_filter_snapshot_for_lora_cohort(
            snapshot, preferred_request_id=str(ingress[0].long_req_id) if ingress else None
        )
        if lora_enabled
        else snapshot
    )
    if ingress is not None:
        cohort = _phase1_ingress_cohort(state, ingress, selected)
    elif _need_wave_slice([remaining for (_, remaining) in snapshot], state.policy):
        cohort = _phase1_live_cohort_from_snapshot(selected, request_id_getter=_safe_request_id)
        if cohort is None and selected is not snapshot:
            cohort = _phase1_live_cohort_from_snapshot(snapshot, request_id_getter=_safe_request_id)
        cohort = cohort or global_cohort
    else:
        cohort = global_cohort or (
            None if lora_enabled else _phase1_build_ingress_fallback_cohort(state, snapshot)
        )
    long_group = None
    if cohort is not None:
        long_group = _phase1_find_seq_group_by_request_id(
            selected, cohort.long_req_id, request_id_getter=_safe_request_id
        )
        if long_group is None and selected is not snapshot:
            long_group = _phase1_find_seq_group_by_request_id(
                snapshot, cohort.long_req_id, request_id_getter=_safe_request_id
            )
    return (cohort, selected, long_group)


def _adapt_phase1_policy(
    state: RuntimeState,
    cohort: Phase1CohortStats,
    waiting: Any,
    *,
    queue_len: int,
    max_wait_us: float,
) -> WaveSlicePolicy | None:
    if not state.policy.phase1_runtime_adaptive_enabled:
        return None
    short_count = _phase1_waiting_short_count(
        waiting, short_threshold_tokens=state.policy.metrics_short_request_tokens
    )
    meta = _phase1_runtime_pressure_meta(
        policy=state.policy,
        cohort=cohort,
        queue_len=queue_len,
        waiting_short_count=short_count,
        max_wait_us=max_wait_us,
        virtual_cap_hit_rate=state.metrics.phase1_virtual_cap_hit_ratio(),
        previous_wall_pressure=state.phase1_runtime_wall_pressure_ema,
    )
    alpha = max(0.0, min(1.0, state.policy.phase1_runtime_ema_alpha))
    for field, key in (
        ("phase1_runtime_wall_pressure_ema", "wall_pressure"),
        ("phase1_runtime_pressure_ema", "effective_pressure"),
    ):
        value = (1.0 - alpha) * float(getattr(state, field)) + alpha * float(meta[key])
        setattr(state, field, value)
        meta[key] = value
    base_policy = state.policy
    (adapted, payload) = _phase1_runtime_adapt_policy(base_policy, meta)
    if adapted is base_policy:
        return None
    state.policy = adapted
    state.phase1_runtime_last_meta = dict(payload)
    state.metrics.record_phase1_runtime_adaptation(
        queue_len=queue_len,
        waiting_short_count=short_count,
        effective_pressure=float(payload["effective_pressure"]),
        wall_pressure=float(payload["wall_pressure"]),
        short_urgency=float(payload["short_urgency"]),
        target_fraction=float(payload["phase1_target_long_fraction"]),
        target_chunk=int(payload["phase1_ingress_target_chunk"]),
    )
    return base_policy


def _choose_phase1_chunk(
    state: RuntimeState,
    cohort: Phase1CohortStats,
    snapshot: list[tuple[Any, int]],
    selected_snapshot: list[tuple[Any, int]],
    long_group: Any,
    *,
    max_wait_us: float,
    queue_len: int,
    scheduler_config: Any,
    original_budget: Any,
    original_threshold: Any,
) -> tuple[int, int | None, str | None]:
    (short_len, long_len) = (cohort.representative_short_len, cohort.long_len)
    adjusted_queue = _phase1_adjusted_queue_len(cohort, queue_len, state.policy)
    baseline = _phase1_baseline_chunk_proxy(
        long_len=long_len,
        original_budget=original_budget,
        original_threshold=original_threshold,
        scheduler_cfg=scheduler_config,
        policy=state.policy,
    )
    ingress = _phase1_find_ingress_virtual_candidate(
        state.phase1_ingress_virtuals, snapshot=snapshot, request_id_getter=_safe_request_id
    )
    eager_chunk = None
    if ingress is not None and state.policy.phase1_ingress_direct_authoritative:
        target = _phase1_cohort_target_len(cohort, state.policy)
        upper = max(short_len + 1, long_len - 1)
        ingress_min = _phase1_effective_ingress_min_chunk(state.policy, target=target)
        eager_target = min(upper, target)
        if upper >= ingress_min:
            eager_target = max(eager_target, ingress_min)
        eager_chunk = _phase1_authoritative_chunk(
            state.policy, state.slicer, target=eager_target, short_len=short_len, upper=upper
        )
        eager_chunk = max(
            _phase1_authoritative_short_floor(
                state.policy, short_len=short_len, target=eager_target
            ),
            min(eager_chunk, upper),
        )
        direct = _phase1_direct_chunk_candidate(
            state=state,
            cohort=cohort,
            total_len=max(1, int(_safe_total_tokens(long_group) or long_len)),
            done_offset=max(0, int((_safe_total_tokens(long_group) or long_len) - long_len)),
            remaining_len=long_len,
            baseline_chunk=baseline,
        )
        state.phase1_virtual_token_caps[str(cohort.long_req_id)] = int(direct or eager_chunk)
    best = state.slicer.choose_dynamic_chunk(
        short_len=short_len,
        long_len=long_len,
        scheduler=state.brain,
        t_wait_us=max_wait_us,
        queue_length=adjusted_queue,
        baseline_chunk=baseline,
    )
    explicit = _phase1_explicit_chunk_from_plan(
        state=state,
        cohort=cohort,
        snapshot=selected_snapshot,
        t_wait_us=max_wait_us,
        queue_length=adjusted_queue,
        baseline_chunk=baseline,
        total_tokens_getter=_safe_total_tokens,
        request_id_getter=_safe_request_id,
    )
    kind = None
    if explicit is not None:
        (best, kind) = explicit
    elif eager_chunk is not None and eager_chunk < best:
        (best, kind) = (eager_chunk, "ingress_authoritative_cap")
    joint_floor = _phase12_joint_phase1_floor(state=state, snapshot=snapshot, policy=state.policy)
    if joint_floor is not None:
        best = max(best, joint_floor)
    if kind in {"direct_authoritative", "ingress_authoritative_eager"}:
        best = max(best, _phase1_effective_ingress_min_chunk(state.policy, target=best))
    else:
        best = _maybe_force_phase1_chunk(
            cohort=cohort,
            queue_len=queue_len,
            chosen_chunk=best,
            slicer=state.slicer,
            policy=state.policy,
        )
    return (int(best), baseline, kind)


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


def _phase1_schedule_plan(
    *,
    state: RuntimeState,
    scheduler_obj: Any,
    lora_enabled: bool,
    scheduler_config: Any,
    original_budget: Any,
    original_threshold: Any,
) -> Phase1ScheduleDecision | None:
    snapshot, max_wait_us = _collect_live_snapshot(scheduler_obj.waiting, scheduler_obj.running)
    cohort, selected, long_group = _select_phase1_cohort(state, snapshot, lora_enabled=lora_enabled)
    if cohort is None or long_group is None or cohort.long_len <= cohort.representative_short_len:
        state.metrics.record_scheduler_decision(False)
        return None
    queue_len = len(scheduler_obj.waiting) + len(scheduler_obj.running)
    _adapt_phase1_policy(
        state, cohort, scheduler_obj.waiting, queue_len=queue_len, max_wait_us=max_wait_us
    )
    best, baseline, explicit_kind = _choose_phase1_chunk(
        state,
        cohort,
        snapshot,
        selected,
        long_group,
        max_wait_us=max_wait_us,
        queue_len=queue_len,
        scheduler_config=scheduler_config,
        original_budget=original_budget,
        original_threshold=original_threshold,
    )
    state.metrics.record_phase1_probe(
        short_len=cohort.representative_short_len,
        long_len=cohort.long_len,
        baseline_chunk=baseline,
        best_chunk=best,
        queue_len=queue_len,
        wait_us=max_wait_us,
        slice_eligible=best < cohort.long_len,
    )
    if best >= cohort.long_len:
        state.metrics.record_scheduler_decision(False)
        return None
    state.metrics.record_scheduler_decision(True)
    state.phase12_recent_phase1_apply_ttl = max(1, state.policy.phase12_phase2_recent_ttl)
    state.phase12_last_phase1_req_id = str(cohort.long_req_id)
    state.phase12_recent_phase1_chunk = max(1, best)
    state.phase12_recent_phase1_strength = max(
        state.phase12_recent_phase1_strength,
        float(max(0, cohort.long_len - best)) / max(1, cohort.long_len),
    )
    state.metrics.record_phase1_choice(
        chosen_chunk=best, baseline_chunk=baseline, explicit_plan=explicit_kind is not None
    )
    return Phase1ScheduleDecision(
        snapshot,
        selected,
        cohort,
        long_group,
        queue_len,
        max_wait_us,
        best,
        baseline,
        explicit_kind,
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


def _build_scheduler_hook(state: RuntimeState) -> Callable[..., Any]:
    original_schedule = state.original_schedule
    schedule_impl = original_schedule

    @functools.wraps(original_schedule)
    def _wave_schedule_hook(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not _has_running_waiting_queues(self) or self.running is None or self.waiting is None:
            return _run_native_schedule(state, self, schedule_impl, args, kwargs)
        _observe_scheduler_requests(state, list(self.waiting) + list(self.running))
        _phase12_tick_recent_phase1(state)
        _phase12_tick_recent_phase2(state)
        if not state.policy.enable_phase1_scheduler:
            state.metrics.record_scheduler_decision(False)
            return _run_native_schedule(state, self, schedule_impl, args, kwargs)
        lora_enabled = _lora_enabled(self)
        can_threshold = state.policy.enable_phase1_dynamic_threshold and (
            not lora_enabled
            or state.policy.allow_phase1_with_lora
            or state.policy.allow_phase1_threshold_with_lora
        )
        can_budget = state.policy.enable_phase1_budget_guidance and (
            not lora_enabled
            or state.policy.allow_phase1_with_lora
            or state.policy.allow_phase1_budget_with_lora
        )
        if lora_enabled and (not (can_threshold or can_budget)):
            state.metrics.record_scheduler_decision(False)
            return _run_native_schedule(state, self, schedule_impl, args, kwargs)
        _phase1_reorder_scheduler_queues(state, self)
        scheduler_config = getattr(self, "scheduler_config", None)
        original_budget = getattr(scheduler_config, "max_num_batched_tokens", None)
        original_threshold = getattr(scheduler_config, "long_prefill_token_threshold", None)
        with _phase1_schedule_scope(
            state=state,
            scheduler_obj=self,
            lora_enabled=lora_enabled,
            can_threshold=can_threshold,
            can_budget=can_budget,
            scheduler_config=scheduler_config,
            original_budget=original_budget,
            original_threshold=original_threshold,
        ) as decision:
            if decision is None:
                return _run_native_schedule(state, self, schedule_impl, args, kwargs)
            outputs = _post_schedule_cashout(state, schedule_impl(self, *args, **kwargs))
            outputs, _, groups, old_sum, new_sum, delta = _phase1_rewrite_scheduler_outputs(
                outputs=outputs,
                request_id=decision.cohort.long_req_id,
                target_chunk=decision.best_chunk,
            )
            state.metrics.record_phase1_rewrite(
                rewritten_groups=groups,
                old_chunk_sum=old_sum,
                new_chunk_sum=new_sum,
                token_delta_sum=delta,
            )
            return outputs

    _wave_schedule_hook.__wave_slice_hook__ = True
    return _wave_schedule_hook
