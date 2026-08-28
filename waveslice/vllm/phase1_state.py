"""Stateful Phase-I ingress and explicit-plan handling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from waveslice.scheduling.slicer import SlicePlan
from waveslice.vllm.phase1_math import (
    need_wave_slice,
    phase1_authoritative_chunk,
    phase1_authoritative_short_floor,
    phase1_cohort_target_len,
    phase1_effective_ingress_min_chunk,
)
from waveslice.vllm.phase1_selection import (
    phase1_find_seq_group_by_request_id,
    phase1_prune_explicit_plans,
)
from waveslice.vllm.state import Phase1CohortStats, Phase1IngressVirtualSlice, RuntimeState

RequestIdGetter = Callable[[Any], str | None]
TotalTokensGetter = Callable[[Any], int | None]


def phase1_maybe_seed_ingress_virtual(
    state: RuntimeState, *, request_id: str, input_tokens: int | None
) -> None:
    request_id = str(request_id)
    if not input_tokens or input_tokens <= 0:
        return
    state.phase1_active_prompt_tokens[request_id] = int(input_tokens)
    active = [
        (rid, int(tokens))
        for rid, tokens in state.phase1_active_prompt_tokens.items()
        if int(tokens) > 0
    ]
    if len(active) < 2:
        return
    lengths = sorted(tokens for _, tokens in active)
    if not need_wave_slice(lengths, state.policy):
        return
    short, long = lengths[0], lengths[-1]
    long_id = next(rid for rid, tokens in active if tokens == long)
    short_lengths = lengths[:-1] or [short]
    virtual = Phase1IngressVirtualSlice(
        long_id,
        short,
        max(1, len(short_lengths)),
        sum(short_lengths),
        short_lengths,
        long,
        len(active),
    )
    state.phase1_ingress_virtuals[long_id] = virtual
    if not state.policy.phase1_ingress_direct_authoritative:
        return
    cohort = Phase1CohortStats(
        short,
        virtual.short_count,
        virtual.short_token_mass,
        short_lengths,
        long,
        long_id,
        len(active),
    )
    cap = phase1_authoritative_chunk(
        state.policy,
        state.slicer,
        target=phase1_cohort_target_len(cohort, state.policy),
        short_len=short,
        upper=long - 1,
    )
    plans = phase1_build_direct_explicit_plans(
        state=state,
        cohort=cohort,
        total_len=long,
        done_offset=0,
        remaining_len=long,
        baseline_chunk=None,
    )
    if plans:
        cap = min(cap, plans[0].chunk_len)
    state.phase1_virtual_token_caps[long_id] = max(1, cap)
    state.metrics.record_phase1_virtual_cap_probe(target_set=True)


def phase1_build_direct_explicit_plans(
    *,
    state: RuntimeState,
    cohort: Phase1CohortStats,
    total_len: int,
    done_offset: int,
    remaining_len: int,
    baseline_chunk: int | None,
) -> list[SlicePlan]:
    if not state.policy.enable_phase1_direct_explicit_override:
        return []
    short, long = (
        max(1, cohort.representative_short_len),
        max(cohort.representative_short_len + 1, remaining_len),
    )
    current = Phase1CohortStats(
        short,
        cohort.short_count,
        cohort.short_token_mass,
        list(cohort.short_lengths),
        long,
        cohort.long_req_id,
        cohort.total_count,
    )
    target = phase1_cohort_target_len(current, state.policy)
    upper = min(long - 1, baseline_chunk - 1) if baseline_chunk and baseline_chunk > 0 else long - 1
    if upper <= short:
        return []
    minimum = phase1_effective_ingress_min_chunk(state.policy, target=target)
    target = min(max(target, minimum) if upper >= minimum else target, upper)
    chunk = phase1_authoritative_chunk(
        state.policy, state.slicer, target=target, short_len=short, upper=upper
    )
    chunk = max(
        phase1_authoritative_short_floor(state.policy, short_len=short, target=target),
        min(chunk, upper),
    )
    if chunk >= long:
        return []
    return [
        state.slicer.make_plan(
            short_len=short, long_total_len=total_len, chunk_len=chunk, long_offset=offset
        )
        for offset, _ in state.slicer.iter_long_chunks(
            long_total_len=total_len, chunk_len=chunk, start_offset=done_offset
        )
    ]


def phase1_direct_chunk_candidate(**kwargs: Any) -> int | None:
    plans = phase1_build_direct_explicit_plans(**kwargs)
    return plans[0].chunk_len if plans else None


def phase1_explicit_chunk_from_plan(
    *,
    state: RuntimeState,
    cohort: Phase1CohortStats,
    snapshot: list[tuple[Any, int]],
    t_wait_us: float,
    queue_length: int,
    baseline_chunk: int | None,
    total_tokens_getter: TotalTokensGetter,
    request_id_getter: RequestIdGetter,
) -> tuple[int, str] | None:
    request_id = cohort.long_req_id
    if not state.policy.enable_phase1_explicit_plan or not request_id:
        return None
    group = phase1_find_seq_group_by_request_id(
        snapshot, request_id, request_id_getter=request_id_getter
    )
    total = total_tokens_getter(group) if group is not None else None
    remaining = next((value for candidate, value in snapshot if candidate is group), 0)
    if total is None or remaining <= 0:
        state.phase1_explicit_plans.pop(request_id, None)
        return None
    offset = max(0, total - remaining)
    plans = phase1_prune_explicit_plans(state.phase1_explicit_plans.get(request_id, []), offset)
    kind = "reuse" if plans else "new"
    if not plans:
        plans = state.slicer.build_long_prefill_plan(
            short_len=cohort.representative_short_len,
            long_total_len=total,
            scheduler=state.brain,
            t_wait_us=max(0.0, t_wait_us),
            queue_length=max(0, queue_length),
            start_offset=offset,
            baseline_chunk=baseline_chunk,
        )
    direct = phase1_build_direct_explicit_plans(
        state=state,
        cohort=cohort,
        total_len=total,
        done_offset=offset,
        remaining_len=remaining,
        baseline_chunk=baseline_chunk,
    )
    if direct:
        direct_chunk, current = direct[0].chunk_len, plans[0].chunk_len if plans else remaining
        minimum = phase1_effective_ingress_min_chunk(state.policy, target=direct_chunk)
        if (
            state.policy.phase1_ingress_direct_authoritative
            and request_id in state.phase1_ingress_virtuals
        ):
            plans, kind = direct, "direct_authoritative"
        elif direct_chunk < current or current < minimum <= direct_chunk:
            plans, kind = direct, "direct"
    if not plans:
        return None
    state.phase1_explicit_plans[request_id] = plans
    return max(1, min(plans[0].chunk_len, remaining)), kind
