"""Select short prefills that benefit from a Phase-II cashout window."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from waveslice.policy import WaveSlicePolicy
from waveslice.vllm.state import Phase12BeneficiarySignal, RuntimeState, ScheduledRequestInfo


def phase12_collect_prefill_lora_state(
    seq_groups: list[Any],
    *,
    rank_infer: Callable[[Any], int],
    remaining_getter: Callable[[Any], int | None],
):
    pairs = [
        (int(remaining), max(1, rank_infer(getattr(group, "lora_request", None)) or 1))
        for group in seq_groups
        if (remaining := remaining_getter(group)) and remaining > 0
    ]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _metric_fields(metric: dict[str, Any], policy: WaveSlicePolicy):
    tokens, arrival, short = (
        metric.get(name) for name in ("input_tokens", "arrival_s", "is_short")
    )
    if short is None and tokens is not None:
        short = int(tokens) <= policy.metrics_short_request_tokens
    return (
        None if tokens is None else int(tokens),
        None if arrival is None else float(arrival),
        bool(short),
    )


def phase12_collect_snapshot_req_infos(
    snapshot: list[tuple[Any, int]],
    *,
    state: RuntimeState | None,
    policy: WaveSlicePolicy,
    request_id_getter: Callable[[Any], str | None],
    expected_chunk_getter: Callable[[Any, RuntimeState | None, int], int],
    rank_infer: Callable[[Any], int],
) -> list[ScheduledRequestInfo]:
    request_ids = [
        str(request_id) for group, _ in snapshot if (request_id := request_id_getter(group))
    ]
    metrics = state.metrics.snapshot_requests(request_ids) if state is not None else {}
    infos = []
    for group, remaining in snapshot:
        request_id = request_id_getter(group)
        if not request_id:
            continue
        request_id, remaining = str(request_id), max(0, int(remaining))
        input_tokens, arrival, short = _metric_fields(metrics.get(request_id, {}), policy)
        infos.append(
            ScheduledRequestInfo(
                request_id,
                remaining,
                remaining,
                expected_chunk_getter(group, state, remaining),
                input_tokens,
                arrival,
                short,
                max(1, rank_infer(getattr(group, "lora_request", None)) or 1),
            )
        )
    return infos


def _empty_signal() -> Phase12BeneficiarySignal:
    return Phase12BeneficiarySignal(None, 0.0, 0.0, [], {})


def _score_beneficiaries(
    *,
    anchor_id: str,
    candidates: list[ScheduledRequestInfo],
    beneficiaries: list[ScheduledRequestInfo],
    target_chunk: int,
    upper: int,
    policy: WaveSlicePolicy,
) -> Phase12BeneficiarySignal:
    fraction = len(beneficiaries) / len(candidates) if candidates else 0.0
    if not beneficiaries:
        return Phase12BeneficiarySignal(anchor_id, fraction, 0.0, [], {})
    now = time.perf_counter()
    waits = [
        max(0.0, now - info.arrival_s) if info.arrival_s is not None else 0.0
        for info in beneficiaries
    ]
    max_wait = max(waits, default=0.0)
    scores = {}
    for info, wait in zip(beneficiaries, waits, strict=False):
        wait_quality = min(1.0, wait / max_wait) if max_wait else 0.0
        if info.remaining_tokens <= target_chunk:
            size_quality = 1.0
        elif info.remaining_tokens >= upper:
            size_quality = 0.0
        else:
            size_quality = 1 - (info.remaining_tokens - target_chunk) / max(1, upper - target_chunk)
        scores[info.request_id] = 0.4 * wait_quality + 0.6 * size_quality
    threshold = max(0.0, min(1.0, policy.phase12_phase2_beneficiary_score_threshold))
    eligible = [info.request_id for info in beneficiaries if scores[info.request_id] >= threshold]
    selected = sorted(eligible, key=scores.get, reverse=True)[:1]
    return Phase12BeneficiarySignal(
        anchor_id,
        fraction,
        scores[selected[0]] if selected else 0.0,
        selected,
        scores,
    )


def phase12_beneficiary_signal(
    *, state: RuntimeState, policy: WaveSlicePolicy, req_infos: list[ScheduledRequestInfo]
) -> Phase12BeneficiarySignal:
    recent_ttl = int(state.phase12_recent_phase1_apply_ttl or 0)
    recent_chunk = max(0, int(state.phase12_recent_phase1_chunk or 0))
    if not req_infos or recent_ttl <= 0 or recent_chunk <= 0:
        return _empty_signal()
    info_map = {info.request_id: info for info in req_infos}
    anchor_id = str(state.phase12_last_phase1_req_id or "")
    anchor = info_map.get(anchor_id)
    if anchor is None:
        prefills = [info for info in req_infos if info.remaining_tokens > 1]
        if not prefills:
            return _empty_signal()
        anchor = max(prefills, key=lambda info: (info.remaining_tokens, info.scheduled_tokens))
        anchor_id = anchor.request_id
    upper = max(
        recent_chunk + 64,
        int(recent_chunk * max(1.0, policy.phase12_phase2_beneficiary_prefill_scale)),
    )
    candidates = [
        info for info in req_infos if info.request_id != anchor_id and info.remaining_tokens > 1
    ]
    beneficiaries = [
        info
        for info in candidates
        if info.remaining_tokens <= upper
        and (
            anchor.arrival_s is None or info.arrival_s is None or info.arrival_s >= anchor.arrival_s
        )
        and (info.is_short or (info.input_tokens is not None and info.input_tokens <= upper))
    ]
    return _score_beneficiaries(
        anchor_id=anchor_id,
        candidates=candidates,
        beneficiaries=beneficiaries,
        target_chunk=recent_chunk,
        upper=upper,
        policy=policy,
    )


def phase2_beneficiary_signal(
    *,
    policy: WaveSlicePolicy,
    req_infos: list[ScheduledRequestInfo],
) -> Phase12BeneficiarySignal:
    prefills = [info for info in req_infos if info.remaining_tokens > 1]
    if len(prefills) < 2:
        return _empty_signal()

    def service(info: ScheduledRequestInfo) -> int:
        return max(1, info.remaining_tokens) * max(1, info.lora_rank)

    anchor = max(prefills, key=lambda info: (service(info), info.remaining_tokens, info.request_id))
    candidates = [info for info in prefills if info.request_id != anchor.request_id]
    cutoff = service(anchor) / max(1.0, policy.phase2_min_hetero_ratio)
    beneficiaries = [info for info in candidates if service(info) <= cutoff]
    if not beneficiaries:
        return Phase12BeneficiarySignal(anchor.request_id, 0.0, 0.0, [], {})
    target = min(info.remaining_tokens for info in beneficiaries)
    upper = max(target + 64, max(info.remaining_tokens for info in beneficiaries))
    return _score_beneficiaries(
        anchor_id=anchor.request_id,
        candidates=candidates,
        beneficiaries=beneficiaries,
        target_chunk=target,
        upper=upper,
        policy=policy,
    )
