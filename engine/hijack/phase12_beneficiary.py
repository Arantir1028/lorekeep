from __future__ import annotations

import time
from typing import Any, Callable, Optional

from engine.hijack.runtime_state import WaveSlicePolicy
from engine.hijack.types import (
    _PatchState,
    _Phase12BeneficiarySignal,
    _ScheduledReqInfo,
)

RankInfer = Callable[[Any], int]
RequestIdGetter = Callable[[Any], Optional[str]]
ExpectedChunkGetter = Callable[[Any, Optional[_PatchState], int], int]


def phase12_collect_prefill_lora_state(
    seq_groups: list[Any],
    *,
    rank_infer: RankInfer,
    remaining_getter: Callable[[Any], Optional[int]],
) -> tuple[list[int], list[int]]:
    prefill_lens: list[int] = []
    lora_ranks: list[int] = []
    for seq_group in seq_groups:
        remaining = remaining_getter(seq_group) or 0
        if remaining <= 0:
            continue
        prefill_lens.append(int(remaining))
        rank = max(1, rank_infer(getattr(seq_group, "lora_request", None)) or 1)
        lora_ranks.append(int(rank))
    return prefill_lens, lora_ranks


def phase12_collect_scheduled_req_infos(
    model_input: Any,
    *,
    runner_self: Optional[Any],
    state: Optional[_PatchState],
    policy: WaveSlicePolicy,
    rank_infer: RankInfer,
) -> list[_ScheduledReqInfo]:
    try:
        num_sched = dict(getattr(model_input, "num_scheduled_tokens", {}) or {})
    except Exception:
        return []
    if not num_sched:
        return []

    req_states = getattr(runner_self, "requests", {}) if runner_self is not None else {}
    metric_snapshot = (
        state.metrics.snapshot_requests(num_sched.keys())
        if state is not None
        else {}
    )
    infos: list[_ScheduledReqInfo] = []
    for rid_raw, tok_raw in num_sched.items():
        rid = str(rid_raw)
        scheduled = max(0, int(tok_raw))
        remaining = scheduled
        rank = 1
        st = req_states.get(rid) if isinstance(req_states, dict) else None
        if st is not None:
            try:
                remaining = max(0, int(st.num_tokens) - int(st.num_computed_tokens))
            except Exception:
                remaining = scheduled
            rank = max(1, rank_infer(getattr(st, "lora_request", None)) or 1)
        metric = metric_snapshot.get(rid, {})
        input_tokens = metric.get("input_tokens")
        arrival_s = metric.get("arrival_s")
        metric_is_short = metric.get("is_short")
        is_short = bool(metric_is_short)
        if metric_is_short is None and input_tokens is not None:
            is_short = int(input_tokens) <= int(policy.metrics_short_request_tokens)
        infos.append(
            _ScheduledReqInfo(
                request_id=rid,
                scheduled_tokens=scheduled,
                remaining_tokens=max(scheduled, remaining),
                expected_chunk_tokens=max(
                    1,
                    min(
                        max(scheduled, remaining),
                        scheduled if scheduled > 0 else max(scheduled, remaining),
                    ),
                ),
                input_tokens=int(input_tokens) if input_tokens is not None else None,
                arrival_s=float(arrival_s) if arrival_s is not None else None,
                is_short=is_short,
                lora_rank=rank,
            )
        )
    return infos


def phase12_collect_snapshot_req_infos(
    snapshot: list[tuple[Any, int]],
    *,
    state: Optional[_PatchState],
    policy: WaveSlicePolicy,
    request_id_getter: RequestIdGetter,
    expected_chunk_getter: ExpectedChunkGetter,
    rank_infer: RankInfer,
) -> list[_ScheduledReqInfo]:
    if not snapshot:
        return []
    request_ids = [
        str(request_id_getter(seq_group))
        for seq_group, _ in snapshot
        if request_id_getter(seq_group)
    ]
    metric_snapshot = (
        state.metrics.snapshot_requests(request_ids)
        if state is not None and request_ids
        else {}
    )
    infos: list[_ScheduledReqInfo] = []
    for seq_group, remaining in snapshot:
        rid = request_id_getter(seq_group)
        if not rid:
            continue
        metric = metric_snapshot.get(str(rid), {})
        input_tokens = metric.get("input_tokens")
        arrival_s = metric.get("arrival_s")
        metric_is_short = metric.get("is_short")
        is_short = bool(metric_is_short)
        if metric_is_short is None and input_tokens is not None:
            is_short = int(input_tokens) <= int(policy.metrics_short_request_tokens)
        rem = max(0, int(remaining))
        infos.append(
            _ScheduledReqInfo(
                request_id=str(rid),
                scheduled_tokens=rem,
                remaining_tokens=rem,
                expected_chunk_tokens=expected_chunk_getter(seq_group, state, rem),
                input_tokens=int(input_tokens) if input_tokens is not None else None,
                arrival_s=float(arrival_s) if arrival_s is not None else None,
                is_short=is_short,
                lora_rank=max(1, rank_infer(getattr(seq_group, "lora_request", None)) or 1),
            )
        )
    return infos


def phase12_beneficiary_signal(
    *,
    state: _PatchState,
    policy: WaveSlicePolicy,
    req_infos: list[_ScheduledReqInfo],
) -> _Phase12BeneficiarySignal:
    if not req_infos:
        return _Phase12BeneficiarySignal(None, [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, [], {})
    recent_ttl = int(getattr(state, "phase12_recent_phase1_apply_ttl", 0) or 0)
    recent_chunk = max(0, int(getattr(state, "phase12_recent_phase1_chunk", 0) or 0))
    if recent_ttl <= 0 or recent_chunk <= 0:
        return _Phase12BeneficiarySignal(None, [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, [], {})

    anchor_id = str(getattr(state, "phase12_last_phase1_req_id", "") or "")
    req_map = {info.request_id: info for info in req_infos}
    anchor = req_map.get(anchor_id)
    if anchor is None:
        prefill_infos = [info for info in req_infos if info.remaining_tokens > 1]
        if not prefill_infos:
            return _Phase12BeneficiarySignal(None, [], 0, 0.0, 0.0, 0.0, 0.0, 0.0, [], {})
        anchor = max(prefill_infos, key=lambda info: (info.remaining_tokens, info.scheduled_tokens))
        anchor_id = anchor.request_id

    anchor_arrival = anchor.arrival_s
    now_s = time.perf_counter()
    beneficiary_upper = max(
        recent_chunk + 64,
        int(float(recent_chunk) * max(1.0, float(policy.phase12_phase2_beneficiary_prefill_scale))),
    )
    beneficiary_ids: list[str] = []
    beneficiary_scores: dict[str, float] = {}
    candidate_pool = 0
    candidate_waits: list[float] = []
    for info in req_infos:
        if info.request_id == anchor_id:
            continue
        if info.remaining_tokens <= 1:
            continue
        candidate_pool += 1
        size_ok = info.remaining_tokens <= beneficiary_upper
        arrival_ok = (
            anchor_arrival is None
            or info.arrival_s is None
            or float(info.arrival_s) >= float(anchor_arrival)
        )
        shortish_ok = info.is_short or (
            info.input_tokens is not None and int(info.input_tokens) <= beneficiary_upper
        )
        if size_ok and arrival_ok and shortish_ok:
            beneficiary_ids.append(info.request_id)
            wait_s = max(0.0, now_s - float(info.arrival_s)) if info.arrival_s is not None else 0.0
            candidate_waits.append(wait_s)
    beneficiary_fraction = (
        float(len(beneficiary_ids)) / float(candidate_pool)
        if candidate_pool > 0
        else 0.0
    )
    if beneficiary_ids:
        max_wait = max(candidate_waits) if candidate_waits else 0.0
        selected_ids: list[str] = []
        wait_scores: list[float] = []
        size_scores: list[float] = []
        score_threshold = max(
            0.0,
            min(1.0, float(getattr(policy, "phase12_phase2_beneficiary_score_threshold", 0.55) or 0.55)),
        )
        wait_weight = max(0.0, float(getattr(policy, "phase12_phase2_beneficiary_wait_weight", 0.40) or 0.40))
        size_weight = max(0.0, float(getattr(policy, "phase12_phase2_beneficiary_size_weight", 0.60) or 0.60))
        norm = max(1e-6, wait_weight + size_weight)
        for rid in beneficiary_ids:
            info = req_map.get(rid)
            if info is None:
                continue
            wait_s = max(0.0, now_s - float(info.arrival_s)) if info.arrival_s is not None else 0.0
            wait_quality = min(1.0, wait_s / max(1e-6, max_wait)) if max_wait > 0.0 else 0.0
            if info.remaining_tokens <= recent_chunk:
                size_quality = 1.0
            elif info.remaining_tokens >= beneficiary_upper:
                size_quality = 0.0
            else:
                size_quality = 1.0 - (
                    float(info.remaining_tokens - recent_chunk)
                    / float(max(1, beneficiary_upper - recent_chunk))
                )
            score = ((wait_weight * wait_quality) + (size_weight * size_quality)) / norm
            beneficiary_scores[rid] = score
            wait_scores.append(wait_quality)
            size_scores.append(size_quality)
            if score >= score_threshold:
                selected_ids.append(rid)
        selected_ids.sort(key=lambda rid: beneficiary_scores.get(rid, 0.0), reverse=True)
        max_selected = max(0, int(getattr(policy, "phase12_phase2_beneficiary_max_selected", 2) or 0))
        if max_selected > 0:
            selected_ids = selected_ids[:max_selected]
        beneficiary_wait_quality = sum(wait_scores) / float(len(wait_scores)) if wait_scores else 0.0
        beneficiary_size_quality = sum(size_scores) / float(len(size_scores)) if size_scores else 0.0
        beneficiary_cashout_quality = (
            sum(beneficiary_scores.get(rid, 0.0) for rid in beneficiary_scores)
            / float(len(beneficiary_scores))
            if beneficiary_scores
            else 0.0
        )
        beneficiary_selected_quality = (
            sum(beneficiary_scores.get(rid, 0.0) for rid in selected_ids)
            / float(len(selected_ids))
            if selected_ids
            else 0.0
        )
    else:
        selected_ids = []
        beneficiary_wait_quality = 0.0
        beneficiary_size_quality = 0.0
        beneficiary_cashout_quality = 0.0
        beneficiary_selected_quality = 0.0
    return _Phase12BeneficiarySignal(
        long_anchor_id=anchor_id,
        beneficiary_prefill_ids=beneficiary_ids,
        beneficiary_prefill_count=len(beneficiary_ids),
        beneficiary_fraction=beneficiary_fraction,
        beneficiary_wait_quality=beneficiary_wait_quality,
        beneficiary_size_quality=beneficiary_size_quality,
        beneficiary_cashout_quality=beneficiary_cashout_quality,
        beneficiary_selected_quality=beneficiary_selected_quality,
        beneficiary_selected_ids=selected_ids,
        beneficiary_score_map=dict(beneficiary_scores),
    )
