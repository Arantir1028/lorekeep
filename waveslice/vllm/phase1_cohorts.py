"""Phase I cohort discovery and per-LoRA candidate selection."""

from __future__ import annotations

from typing import Any

from waveslice.vllm.common import (
    infer_lora_rank as _infer_lora_rank,
    safe_first_seq as _safe_first_seq,
    safe_lora_path as _safe_lora_path,
    safe_request_id as _safe_request_id,
    safe_total_tokens as _safe_total_tokens,
)
from waveslice.vllm.phase1_math import (
    need_wave_slice as _need_wave_slice,
    phase1_adjusted_queue_len as _phase1_adjusted_queue_len,
    phase1_baseline_chunk_proxy as _phase1_baseline_chunk_proxy,
    phase1_cohort_target_len as _phase1_cohort_target_len,
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
from waveslice.vllm.state import Phase1CohortStats, Phase1IngressVirtualSlice, RuntimeState


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
