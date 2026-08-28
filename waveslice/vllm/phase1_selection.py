from __future__ import annotations

from collections.abc import Callable
from typing import Any

from waveslice.scheduling.slicer import SlicePlan
from waveslice.vllm.state import Phase1CohortStats, Phase1IngressVirtualSlice

RequestIdGetter = Callable[[Any], str | None]


def phase1_live_cohort_from_snapshot(
    snapshot: list[tuple[Any, int]], *, request_id_getter: RequestIdGetter
) -> Phase1CohortStats | None:
    positive = sorted(int(rem) for (_, rem) in snapshot if int(rem) > 0)
    if len(positive) < 2:
        return None
    short_len = int(positive[0])
    long_len = int(positive[-1])
    long_req_id = None
    for seq_group, rem in snapshot:
        if int(rem) == long_len:
            long_req_id = request_id_getter(seq_group)
            break
    return Phase1CohortStats(
        representative_short_len=short_len,
        short_count=1,
        short_token_mass=short_len,
        short_lengths=[short_len],
        long_len=long_len,
        long_req_id=long_req_id,
        total_count=len(positive),
    )


def phase1_find_ingress_virtual_candidate(
    ingress_virtuals: dict[str, Phase1IngressVirtualSlice],
    *,
    snapshot: list[tuple[Any, int]],
    request_id_getter: RequestIdGetter,
) -> tuple[Phase1IngressVirtualSlice, Any, int] | None:
    if not ingress_virtuals:
        return None
    best: tuple[Phase1IngressVirtualSlice, Any, int] | None = None
    best_score: tuple[int, int, int] | None = None
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
        score = (
            int(rem),
            int(getattr(candidate, "original_long_len", rem) or rem),
            int(getattr(candidate, "active_count", 0) or 0),
        )
        if best_score is None or score > best_score:
            best = (candidate, seq_group, rem)
            best_score = score
    return best


def phase1_find_seq_group_by_request_id(
    snapshot: list[tuple[Any, int]],
    request_id: str | None,
    *,
    request_id_getter: RequestIdGetter,
) -> Any | None:
    if not request_id:
        return None
    for seq_group, _ in snapshot:
        if request_id_getter(seq_group) == request_id:
            return seq_group
    return None


def phase1_prune_explicit_plans(plans: list[SlicePlan], current_offset: int) -> list[SlicePlan]:
    return [plan for plan in plans if int(plan.long_offset + plan.chunk_len) > int(current_offset)]
