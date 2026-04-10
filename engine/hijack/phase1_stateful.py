from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from engine.base_slicer import SlicePlan, WaveBaseSlicer
from engine.hijack.phase1_math import (
    need_wave_slice,
    phase1_authoritative_chunk,
    phase1_authoritative_short_floor,
    phase1_cohort_target_len,
    phase1_effective_ingress_min_chunk,
)
from engine.hijack.phase1_selection import (
    phase1_find_seq_group_by_request_id,
    phase1_prune_explicit_plans,
)
from engine.hijack.types import (
    _PatchState,
    _Phase1CohortStats,
    _Phase1IngressVirtualSlice,
)

logger = logging.getLogger("WaveSlice")

RequestIdGetter = Callable[[Any], Optional[str]]
TotalTokensGetter = Callable[[Any], Optional[int]]


def phase1_maybe_seed_ingress_virtual(
    state: _PatchState,
    *,
    request_id: str,
    input_tokens: Optional[int],
) -> None:
    if input_tokens is None or int(input_tokens) <= 0:
        state.metrics.record_phase1_step_trace(
            request_id=str(request_id),
            event="phase1_ingress_skip_no_input_tokens",
            is_prefill=True,
            num_computed_tokens=0,
            uncached=(int(input_tokens) if input_tokens is not None else None),
            cached=0,
        )
        return
    state.phase1_active_prompt_tokens[str(request_id)] = int(input_tokens)
    state.metrics.record_phase1_step_trace(
        request_id=str(request_id),
        event="phase1_ingress_candidate_seen",
        is_prefill=True,
        num_computed_tokens=0,
        uncached=int(input_tokens),
        cached=0,
    )
    positive_items = [
        (rid, int(tok))
        for rid, tok in state.phase1_active_prompt_tokens.items()
        if int(tok) > 0
    ]
    if len(positive_items) < 2:
        state.metrics.record_phase1_step_trace(
            request_id=str(request_id),
            event="phase1_ingress_skip_need_pair",
            is_prefill=True,
            num_computed_tokens=0,
            uncached=int(input_tokens),
            cached=0,
        )
        return
    positive = sorted(tok for _, tok in positive_items)
    if not need_wave_slice(positive, state.policy):
        state.metrics.record_phase1_step_trace(
            request_id=str(request_id),
            event="phase1_ingress_skip_no_need_wave_slice",
            is_prefill=True,
            num_computed_tokens=0,
            uncached=int(max(positive)),
            cached=int(min(positive)),
        )
        return
    long_len = positive[-1]
    short_len = positive[0]
    long_req_id = None
    for rid, tok in positive_items:
        if int(tok) == int(long_len):
            long_req_id = str(rid)
            break
    if long_req_id is None:
        state.metrics.record_phase1_step_trace(
            request_id=str(request_id),
            event="phase1_ingress_skip_no_long_req",
            is_prefill=True,
            num_computed_tokens=0,
            uncached=int(long_len),
            cached=int(short_len),
        )
        return
    state.phase1_ingress_virtuals[long_req_id] = _Phase1IngressVirtualSlice(
        long_req_id=long_req_id,
        representative_short_len=int(short_len),
        short_count=max(1, len(positive) - 1),
        short_token_mass=int(sum(positive[:-1])) if len(positive) > 1 else int(short_len),
        short_lengths=[int(v) for v in positive[:-1]] if len(positive) > 1 else [int(short_len)],
        original_long_len=int(long_len),
        active_count=len(positive_items),
    )
    state.metrics.record_phase1_step_trace(
        request_id=str(long_req_id),
        event="phase1_ingress_seed",
        is_prefill=True,
        num_computed_tokens=0,
        uncached=int(long_len),
        cached=int(short_len),
    )
    if bool(state.policy.phase1_ingress_direct_authoritative):
        seed_cohort = _Phase1CohortStats(
            representative_short_len=int(short_len),
            short_count=max(1, len(positive) - 1),
            short_token_mass=int(sum(positive[:-1])) if len(positive) > 1 else int(short_len),
            short_lengths=[int(v) for v in positive[:-1]] if len(positive) > 1 else [int(short_len)],
            long_len=int(long_len),
            long_req_id=str(long_req_id),
            total_count=len(positive_items),
        )
        ingress_target = int(phase1_cohort_target_len(seed_cohort, state.policy))
        ingress_cap = phase1_authoritative_chunk(
            state.policy,
            state.slicer,
            target=int(ingress_target),
            short_len=int(short_len),
            upper=max(1, int(long_len) - 1),
        )
        direct_plans = phase1_build_direct_explicit_plans(
            state=state,
            cohort=seed_cohort,
            total_len=int(long_len),
            done_offset=0,
            remaining_len=int(long_len),
            baseline_chunk=None,
        )
        if direct_plans:
            ingress_cap = min(int(ingress_cap), int(direct_plans[0].chunk_len))
        state.phase1_virtual_token_caps[long_req_id] = max(1, int(ingress_cap))
        state.metrics.record_phase1_virtual_cap_probe(target_set=True)
        state.metrics.record_phase1_step_trace(
            request_id=str(long_req_id),
            event="phase1_ingress_direct_cap_seeded",
            is_prefill=True,
            num_computed_tokens=0,
            uncached=int(long_len),
            cached=int(short_len),
            target_chunk=int(max(1, int(ingress_cap))),
        )
    logger.info(
        "[Wave-Slice][P1-ingress-seed] long_req=%s short=%d long=%d active=%d",
        long_req_id,
        int(short_len),
        int(long_len),
        len(positive_items),
    )


def phase1_apply_sticky_chunk(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    chosen_chunk: int,
    slicer: WaveBaseSlicer,
) -> tuple[int, bool]:
    sticky_req = state.phase1_sticky_req_id
    sticky_chunk = state.phase1_sticky_chunk
    ttl_left = state.phase1_sticky_ttl_left
    if (
        sticky_req
        and sticky_chunk
        and ttl_left > 0
        and cohort.long_req_id
        and sticky_req == cohort.long_req_id
        and cohort.long_len >= int(sticky_chunk * max(1.0, float(state.policy.phase1_sticky_reuse_ratio)))
    ):
        reused = slicer._conservative_map_down(min(int(sticky_chunk), cohort.long_len - 1))
        reused = max(cohort.representative_short_len, min(int(reused), cohort.long_len - 1))
        state.phase1_sticky_ttl_left = max(0, ttl_left - 1)
        return min(chosen_chunk, reused), True

    state.phase1_sticky_req_id = None
    state.phase1_sticky_chunk = None
    state.phase1_sticky_ttl_left = 0
    return chosen_chunk, False


def phase1_update_sticky_chunk(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    chosen_chunk: int,
    applied: bool,
) -> None:
    if applied and cohort.long_req_id:
        state.phase1_sticky_req_id = cohort.long_req_id
        state.phase1_sticky_chunk = int(chosen_chunk)
        state.phase1_sticky_ttl_left = max(0, int(state.policy.phase1_sticky_ttl))
        return
    if state.phase1_sticky_ttl_left > 0:
        state.phase1_sticky_ttl_left = max(0, state.phase1_sticky_ttl_left - 1)
    if state.phase1_sticky_ttl_left == 0:
        state.phase1_sticky_req_id = None
        state.phase1_sticky_chunk = None


def phase1_build_direct_explicit_plans(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    total_len: int,
    done_offset: int,
    remaining_len: int,
    baseline_chunk: Optional[int],
) -> list[SlicePlan]:
    if not bool(state.policy.enable_phase1_direct_explicit_override):
        return []
    short_len = max(1, int(cohort.representative_short_len))
    long_len = max(short_len + 1, int(remaining_len))
    direct_target = phase1_cohort_target_len(
        _Phase1CohortStats(
            representative_short_len=short_len,
            short_count=cohort.short_count,
            short_token_mass=cohort.short_token_mass,
            short_lengths=list(cohort.short_lengths),
            long_len=long_len,
            long_req_id=cohort.long_req_id,
            total_count=cohort.total_count,
        ),
        state.policy,
    )
    upper = long_len - 1
    if baseline_chunk is not None and int(baseline_chunk) > 0:
        upper = min(upper, int(baseline_chunk) - 1)
    if upper <= short_len:
        return []
    ingress_min = phase1_effective_ingress_min_chunk(
        state.policy,
        target=int(direct_target),
    )
    ingress_max = max(ingress_min, int(state.policy.phase1_ingress_max_chunk))
    if upper >= ingress_min:
        direct_target = max(int(direct_target), ingress_min)
    direct_target = min(int(direct_target), upper)
    direct_chunk = phase1_authoritative_chunk(
        state.policy,
        state.slicer,
        target=int(direct_target),
        short_len=int(short_len),
        upper=int(upper),
    )
    direct_floor = phase1_authoritative_short_floor(
        state.policy,
        short_len=int(short_len),
        target=int(direct_target),
    )
    direct_chunk = max(direct_floor, min(int(direct_chunk), upper))
    if direct_chunk >= long_len:
        return []
    return [
        state.slicer.make_plan(
            short_len=short_len,
            long_total_len=int(total_len),
            chunk_len=int(direct_chunk),
            long_offset=int(chunk_offset),
        )
        for chunk_offset, _ in state.slicer.iter_long_chunks(
            long_total_len=int(total_len),
            chunk_len=int(direct_chunk),
            start_offset=int(done_offset),
        )
    ]


def phase1_direct_chunk_candidate(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    total_len: int,
    done_offset: int,
    remaining_len: int,
    baseline_chunk: Optional[int],
) -> Optional[int]:
    plans = phase1_build_direct_explicit_plans(
        state=state,
        cohort=cohort,
        total_len=int(total_len),
        done_offset=int(done_offset),
        remaining_len=int(remaining_len),
        baseline_chunk=baseline_chunk,
    )
    if not plans:
        return None
    return int(plans[0].chunk_len)


def phase1_explicit_chunk_from_plan(
    *,
    state: _PatchState,
    cohort: _Phase1CohortStats,
    snapshot: list[tuple[Any, int]],
    t_wait_us: float,
    queue_length: int,
    baseline_chunk: Optional[int],
    total_tokens_getter: TotalTokensGetter,
    request_id_getter: RequestIdGetter,
) -> Optional[tuple[int, str]]:
    if not bool(state.policy.enable_phase1_explicit_plan):
        return None
    request_id = cohort.long_req_id
    if not request_id:
        return None
    seq_group = phase1_find_seq_group_by_request_id(
        snapshot,
        request_id,
        request_id_getter=request_id_getter,
    )
    if seq_group is None:
        state.phase1_explicit_plans.pop(request_id, None)
        return None

    total_len = total_tokens_getter(seq_group)
    remaining_len = int(next((rem for sg, rem in snapshot if sg is seq_group), 0))
    if total_len is None or remaining_len is None:
        state.phase1_explicit_plans.pop(request_id, None)
        return None
    done_offset = max(0, int(total_len) - int(remaining_len))
    if remaining_len <= 0:
        state.phase1_explicit_plans.pop(request_id, None)
        return None

    existing_plans = list(state.phase1_explicit_plans.get(request_id, []))
    plans = phase1_prune_explicit_plans(existing_plans, done_offset)
    plan_kind = "reuse" if plans else "new"
    if not plans:
        plans = state.slicer.build_long_prefill_plan(
            short_len=int(cohort.representative_short_len),
            long_total_len=int(total_len),
            scheduler=state.brain,
            t_wait_us=float(max(0.0, t_wait_us)),
            queue_length=int(max(0, queue_length)),
            start_offset=int(done_offset),
            baseline_chunk=baseline_chunk,
        )
    direct_plans = phase1_build_direct_explicit_plans(
        state=state,
        cohort=cohort,
        total_len=int(total_len),
        done_offset=int(done_offset),
        remaining_len=int(remaining_len),
        baseline_chunk=baseline_chunk,
    )
    if direct_plans:
        if bool(state.policy.phase1_ingress_direct_authoritative) and request_id in state.phase1_ingress_virtuals:
            plans = direct_plans
            plan_kind = "direct_authoritative"
        else:
            first_direct = direct_plans[0]
            first_plan = plans[0] if plans else None
            current_chunk = int(first_plan.chunk_len) if first_plan is not None else int(remaining_len)
            direct_chunk = int(first_direct.chunk_len)
            ingress_min = phase1_effective_ingress_min_chunk(
                state.policy,
                target=int(direct_chunk),
            )
            if direct_chunk < current_chunk or (current_chunk < ingress_min <= direct_chunk):
                plans = direct_plans
                plan_kind = "direct"
    if not plans:
        state.phase1_explicit_plans.pop(request_id, None)
        return None

    state.phase1_explicit_plans[request_id] = plans
    active = plans[0]
    chunk_len = max(1, min(int(active.chunk_len), int(remaining_len)))
    return chunk_len, plan_kind
