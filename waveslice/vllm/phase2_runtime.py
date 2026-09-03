"""Phase II coordination and scheduler cashout for vLLM V1."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from waveslice.policy import WaveSlicePolicy
from waveslice.vllm.common import (
    collect_live_snapshot as _collect_live_snapshot,
    infer_lora_rank as _infer_lora_rank,
    phase12_expected_chunk_tokens as _phase12_expected_chunk_tokens,
    rebuild_queue_like as _rebuild_queue_like,
    safe_prefill_uncomputed_tokens as _safe_prefill_uncomputed_tokens,
    safe_remaining_tokens as _safe_remaining_tokens,
    safe_request_id as _safe_request_id,
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
from waveslice.vllm.state import (
    Phase12BeneficiarySignal,
    RuntimeState,
    ScheduledRequestInfo,
)


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
