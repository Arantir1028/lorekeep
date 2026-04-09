from __future__ import annotations

from typing import Any, Callable, Optional

from engine.base_slicer import SlicePlan
from engine.hijack.runtime_state import WaveSlicePolicy
from engine.hijack.types import _Phase1CohortStats, _Phase1IngressVirtualSlice


RequestIdGetter = Callable[[Any], Optional[str]]


def phase1_build_cohort(
    snapshot: list[tuple[Any, int]],
    policy: WaveSlicePolicy,
    *,
    request_id_getter: RequestIdGetter,
) -> Optional[_Phase1CohortStats]:
    positive = sorted(int(rem) for _, rem in snapshot if int(rem) > 0)
    if len(positive) < 2:
        return None
    long_len = positive[-1]
    cohort_cap = max(1, int(long_len * float(policy.phase1_short_cohort_long_fraction)))
    short_lengths = [int(v) for v in positive[:-1] if int(v) <= cohort_cap]
    if len(short_lengths) < int(policy.phase1_cohort_min_count):
        short_lengths = positive[:-1]
    if not short_lengths:
        short_lengths = [positive[0]]
    short_count = len(short_lengths)
    short_token_mass = sum(short_lengths)
    representative = max(1, int(round(short_token_mass / max(1, short_count))))
    long_req_id = None
    for seq_group, rem in snapshot:
        if int(rem) == int(long_len):
            long_req_id = request_id_getter(seq_group)
            break
    return _Phase1CohortStats(
        representative_short_len=representative,
        short_count=short_count,
        short_token_mass=short_token_mass,
        short_lengths=list(short_lengths),
        long_len=int(long_len),
        long_req_id=long_req_id,
        total_count=len(positive),
    )


def phase1_basic_cohort(
    snapshot: list[tuple[Any, int]],
    *,
    request_id_getter: RequestIdGetter,
) -> Optional[_Phase1CohortStats]:
    positive = sorted(int(rem) for _, rem in snapshot if int(rem) > 0)
    if len(positive) < 2:
        return None
    short_len = int(positive[0])
    long_len = int(positive[-1])
    long_req_id = None
    for seq_group, rem in snapshot:
        if int(rem) == long_len:
            long_req_id = request_id_getter(seq_group)
            break
    return _Phase1CohortStats(
        representative_short_len=short_len,
        short_count=1,
        short_token_mass=short_len,
        short_lengths=[short_len],
        long_len=long_len,
        long_req_id=long_req_id,
        total_count=len(positive),
    )


def phase1_live_cohort_from_snapshot(
    snapshot: list[tuple[Any, int]],
    policy: WaveSlicePolicy,
    *,
    request_id_getter: RequestIdGetter,
) -> Optional[_Phase1CohortStats]:
    if policy.phase1_enable_cohort_mode:
        return phase1_build_cohort(snapshot, policy, request_id_getter=request_id_getter)
    return phase1_basic_cohort(snapshot, request_id_getter=request_id_getter)


def phase1_find_ingress_virtual_candidate(
    ingress_virtuals: dict[str, _Phase1IngressVirtualSlice],
    *,
    snapshot: list[tuple[Any, int]],
    request_id_getter: RequestIdGetter,
) -> Optional[tuple[_Phase1IngressVirtualSlice, Any, int]]:
    if not ingress_virtuals:
        return None
    for seq_group, remaining in snapshot:
        req_id = request_id_getter(seq_group)
        if not req_id:
            continue
        candidate = ingress_virtuals.get(str(req_id))
        if candidate is None:
            continue
        rem = int(remaining)
        if rem <= 0:
            continue
        return candidate, seq_group, rem
    return None


def phase1_find_seq_group_by_request_id(
    snapshot: list[tuple[Any, int]],
    request_id: Optional[str],
    *,
    request_id_getter: RequestIdGetter,
) -> Optional[Any]:
    if not request_id:
        return None
    for seq_group, _ in snapshot:
        if request_id_getter(seq_group) == request_id:
            return seq_group
    return None


def phase1_prune_explicit_plans(
    plans: list[SlicePlan],
    current_offset: int,
) -> list[SlicePlan]:
    return [
        plan
        for plan in plans
        if int(plan.long_offset + plan.chunk_len) > int(current_offset)
    ]
