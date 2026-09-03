"""Phase I policy adaptation and schedule-plan construction for vLLM V1."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from waveslice.policy import WaveSlicePolicy
from waveslice.vllm.common import (
    collect_live_snapshot as _collect_live_snapshot,
    safe_prefill_uncomputed_tokens as _safe_prefill_uncomputed_tokens,
    safe_request_id as _safe_request_id,
    safe_total_tokens as _safe_total_tokens,
)
from waveslice.vllm.phase1_cohorts import _select_phase1_cohort
from waveslice.vllm.phase1_math import (
    maybe_force_phase1_chunk as _maybe_force_phase1_chunk,
    phase1_adjusted_queue_len as _phase1_adjusted_queue_len,
    phase1_authoritative_chunk as _phase1_authoritative_chunk,
    phase1_authoritative_short_floor as _phase1_authoritative_short_floor,
    phase1_baseline_chunk_proxy as _phase1_baseline_chunk_proxy,
    phase1_cohort_target_len as _phase1_cohort_target_len,
    phase1_effective_ingress_min_chunk as _phase1_effective_ingress_min_chunk,
    phase1_runtime_adapt_policy as _phase1_runtime_adapt_policy,
    phase1_runtime_pressure_meta as _phase1_runtime_pressure_meta,
)
from waveslice.vllm.phase1_selection import (
    phase1_find_ingress_virtual_candidate as _phase1_find_ingress_virtual_candidate,
)
from waveslice.vllm.phase1_state import (
    phase1_direct_chunk_candidate as _phase1_direct_chunk_candidate,
    phase1_explicit_chunk_from_plan as _phase1_explicit_chunk_from_plan,
)
from waveslice.vllm.phase2_runtime import _phase12_joint_phase1_floor
from waveslice.vllm.state import Phase1CohortStats, Phase1ScheduleDecision, RuntimeState


def _phase1_waiting_short_count(waiting: Iterable[Any], *, short_threshold_tokens: int) -> int:
    threshold = max(1, int(short_threshold_tokens))
    return sum(
        1 for group in waiting if 0 < int(_safe_prefill_uncomputed_tokens(group) or 0) <= threshold
    )


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
