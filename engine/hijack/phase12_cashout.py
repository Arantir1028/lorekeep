from __future__ import annotations

import time
from typing import Any, Optional

from engine.hijack.runtime_state import WaveSlicePolicy
from engine.hijack.types import _Phase12BeneficiarySignal, _ScheduledReqInfo


def phase12_scheduler_cashout_grade(
    *,
    policy: WaveSlicePolicy,
    selected_ids: list[str],
    selected_quality: float,
    removable_count: int,
    value_signal: Optional[dict[str, float]] = None,
) -> Optional[dict[str, Any]]:
    if not selected_ids or removable_count <= 0:
        return None
    soft_floor = max(
        0.0,
        min(
            1.0,
            float(getattr(policy, "phase12_phase2_scheduler_cashout_soft_floor", 0.55) or 0.55),
        ),
    )
    hard_floor = max(
        soft_floor + 1e-6,
        min(
            1.0,
            float(
                getattr(policy, "phase12_phase2_scheduler_cashout_quality_floor", 0.78) or 0.78
            ),
        ),
    )
    quality = max(0.0, min(1.0, float(selected_quality)))
    if quality < soft_floor:
        return {
            "allowed": False,
            "soft_floor": soft_floor,
            "hard_floor": hard_floor,
            "strength": 0.0,
            "selected_cap": 0,
            "remove_cap": 0,
            "lane_ttl": 0,
        }
    candidate_size_quality = float((value_signal or {}).get("candidate_size_quality", 0.0) or 0.0)
    candidate_shape_penalty = float((value_signal or {}).get("candidate_shape_penalty", 0.0) or 0.0)
    small_candidate_bonus = float((value_signal or {}).get("small_candidate_bonus", 0.0) or 0.0)
    medium_candidate_penalty = float((value_signal or {}).get("medium_candidate_penalty", 0.0) or 0.0)
    effective_hard_floor = max(
        soft_floor + 1e-6,
        min(
            1.0,
            hard_floor
            - (0.65 * small_candidate_bonus)
            + (0.35 * medium_candidate_penalty),
        ),
    )
    if quality >= effective_hard_floor:
        strength = 1.0
    else:
        strength = (quality - soft_floor) / max(1e-6, effective_hard_floor - soft_floor)
    strength = max(0.0, min(1.0, float(strength)))
    value_score = max(
        0.0,
        min(1.0, float((value_signal or {}).get("value_score", 1.0) or 0.0)),
    )
    net_value = float((value_signal or {}).get("net_value", value_score) or 0.0)
    gain_score = float((value_signal or {}).get("gain_score", value_score) or 0.0)
    cost_score = float((value_signal or {}).get("cost_score", 0.0) or 0.0)
    wait_gap = float((value_signal or {}).get("wait_gap", 0.0) or 0.0)
    candidate_wait_quality = float((value_signal or {}).get("candidate_wait_quality", 0.0) or 0.0)
    if value_score <= 0.10 or net_value <= 0.0:
        return {
            "allowed": False,
            "soft_floor": soft_floor,
            "hard_floor": effective_hard_floor,
            "strength": 0.0,
            "selected_cap": 0,
            "remove_cap": 0,
            "lane_ttl": 0,
            "value_score": value_score,
            "net_value": net_value,
            "gain_score": gain_score,
            "cost_score": cost_score,
            "wait_gap": wait_gap,
            "candidate_wait_quality": candidate_wait_quality,
            "candidate_size_quality": candidate_size_quality,
            "candidate_shape_penalty": candidate_shape_penalty,
            "small_candidate_bonus": small_candidate_bonus,
            "medium_candidate_penalty": medium_candidate_penalty,
        }
    strength = max(0.0, min(1.0, strength * value_score))

    return {
        "allowed": True,
        "soft_floor": soft_floor,
        "hard_floor": effective_hard_floor,
        "strength": strength,
        "selected_cap": 1,
        "remove_cap": 1,
        "lane_ttl": 1,
        "value_score": value_score,
        "net_value": net_value,
        "gain_score": gain_score,
        "cost_score": cost_score,
        "wait_gap": wait_gap,
        "candidate_wait_quality": candidate_wait_quality,
        "candidate_size_quality": candidate_size_quality,
        "candidate_shape_penalty": candidate_shape_penalty,
        "small_candidate_bonus": small_candidate_bonus,
        "medium_candidate_penalty": medium_candidate_penalty,
    }


def phase12_scheduler_cashout_cooldown_for_grade(
    *,
    policy: WaveSlicePolicy,
    grade: Optional[dict[str, Any]],
) -> int:
    max_cooldown = max(
        0,
        int(getattr(policy, "phase12_phase2_scheduler_cashout_cooldown_ticks", 2) or 0),
    )
    if max_cooldown <= 0 or not grade:
        return 0
    strength = max(0.0, min(1.0, float((grade or {}).get("strength", 0.0) or 0.0)))
    value_score = max(0.0, min(1.0, float((grade or {}).get("value_score", 0.0) or 0.0)))
    control_score = (0.5 * strength) + (0.5 * value_score)
    if control_score >= 0.70:
        return 0
    if control_score >= 0.32:
        return min(1, max_cooldown)
    return max_cooldown


def phase12_scheduler_cashout_value_signal(
    *,
    req_infos: list[_ScheduledReqInfo],
    beneficiary_signal: _Phase12BeneficiarySignal,
    removable_request_ids: Optional[set[str]] = None,
    removable_candidate_request_ids: Optional[list[str]] = None,
    removable_candidate_chunk_tokens: Optional[dict[str, int]] = None,
) -> dict[str, float]:
    if not req_infos:
        return {
            "gain_score": 0.0,
            "cost_score": 1.0,
            "value_score": 0.0,
            "net_value": -1.0,
            "removable_wait_quality": 1.0,
            "removable_size_quality": 1.0,
        }
    beneficiary_ids = {
        str(rid)
        for rid in (beneficiary_signal.beneficiary_selected_ids or [])
        if str(rid)
    }
    prefill_infos = [info for info in req_infos if int(info.remaining_tokens) > 1]
    removable_infos = [
        info
        for info in prefill_infos
        if str(info.request_id) not in beneficiary_ids
        and (
            removable_request_ids is None
            or str(info.request_id) in removable_request_ids
        )
    ]
    now_s = time.perf_counter()
    waits = [
        max(0.0, now_s - float(info.arrival_s))
        for info in prefill_infos
        if info.arrival_s is not None
    ]
    max_wait = max(waits) if waits else 0.0
    max_expected_chunk = max((int(info.expected_chunk_tokens) for info in prefill_infos), default=0)

    removable_wait_scores: list[float] = []
    removable_size_scores: list[float] = []
    for info in removable_infos:
        wait_s = max(0.0, now_s - float(info.arrival_s)) if info.arrival_s is not None else 0.0
        wait_quality = min(1.0, wait_s / max(1e-6, max_wait)) if max_wait > 0.0 else 0.0
        size_quality = (
            min(1.0, float(max(0, int(info.expected_chunk_tokens))) / float(max_expected_chunk))
            if max_expected_chunk > 0
            else 0.0
        )
        removable_wait_scores.append(wait_quality)
        removable_size_scores.append(size_quality)

    removable_wait_quality = (
        sum(removable_wait_scores) / float(len(removable_wait_scores))
        if removable_wait_scores
        else 1.0
    )
    removable_size_quality = (
        sum(removable_size_scores) / float(len(removable_size_scores))
        if removable_size_scores
        else 1.0
    )

    removable_map = {str(info.request_id): info for info in removable_infos}
    candidate_infos: list[_ScheduledReqInfo] = []
    for rid in removable_candidate_request_ids or []:
        info = removable_map.get(str(rid))
        if info is not None:
            candidate_infos.append(info)

    candidate_wait_scores: list[float] = []
    candidate_size_scores: list[float] = []
    for info in candidate_infos:
        wait_s = max(0.0, now_s - float(info.arrival_s)) if info.arrival_s is not None else 0.0
        wait_quality = min(1.0, wait_s / max(1e-6, max_wait)) if max_wait > 0.0 else 0.0
        candidate_chunk_tokens = int(
            (removable_candidate_chunk_tokens or {}).get(str(info.request_id), info.expected_chunk_tokens)
            or info.expected_chunk_tokens
            or 0
        )
        size_quality = (
            min(1.0, float(max(0, candidate_chunk_tokens)) / float(max_expected_chunk))
            if max_expected_chunk > 0
            else 0.0
        )
        candidate_wait_scores.append(wait_quality)
        candidate_size_scores.append(size_quality)

    candidate_wait_quality = (
        sum(candidate_wait_scores) / float(len(candidate_wait_scores))
        if candidate_wait_scores
        else removable_wait_quality
    )
    candidate_size_quality = (
        sum(candidate_size_scores) / float(len(candidate_size_scores))
        if candidate_size_scores
        else removable_size_quality
    )

    beneficiary_wait_quality = float(beneficiary_signal.beneficiary_wait_quality)
    wait_gap = max(-1.0, min(1.0, beneficiary_wait_quality - removable_wait_quality))
    positive_wait_gap = max(0.0, wait_gap)
    negative_wait_gap = max(0.0, -wait_gap)
    candidate_shape_penalty = max(
        0.0,
        candidate_size_quality - float(beneficiary_signal.beneficiary_size_quality),
    )
    small_candidate_bonus = 0.0
    if candidate_size_quality <= 0.15:
        small_candidate_bonus = 0.12 * (1.0 - (candidate_size_quality / 0.15))
    medium_candidate_penalty = 0.0
    if candidate_size_quality >= 0.35:
        medium_candidate_penalty = 0.18 * min(
            1.0, (candidate_size_quality - 0.35) / 0.35
        )

    gain_score = (
        0.42 * float(beneficiary_signal.beneficiary_selected_quality)
        + 0.23 * beneficiary_wait_quality
        + 0.10 * float(beneficiary_signal.beneficiary_size_quality)
        + 0.05 * float(beneficiary_signal.beneficiary_fraction)
        + 0.20 * positive_wait_gap
        + small_candidate_bonus
    )
    cost_score = (
        0.25 * removable_wait_quality
        + 0.10 * removable_size_quality
        + 0.15 * candidate_wait_quality
        + 0.30 * candidate_size_quality
        + 0.15 * candidate_shape_penalty
        + 0.05 * negative_wait_gap
        + medium_candidate_penalty
    )
    net_value = float(gain_score) - float(cost_score)
    value_score = max(
        0.0,
        min(1.0, net_value + 0.25),
    )
    return {
        "gain_score": float(gain_score),
        "cost_score": float(cost_score),
        "value_score": float(value_score),
        "net_value": float(net_value),
        "wait_gap": float(wait_gap),
        "candidate_wait_quality": float(candidate_wait_quality),
        "candidate_size_quality": float(candidate_size_quality),
        "candidate_shape_penalty": float(candidate_shape_penalty),
        "small_candidate_bonus": float(small_candidate_bonus),
        "medium_candidate_penalty": float(medium_candidate_penalty),
        "removable_wait_quality": float(removable_wait_quality),
        "removable_size_quality": float(removable_size_quality),
    }
