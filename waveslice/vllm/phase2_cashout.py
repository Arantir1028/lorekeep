"""Choose and grade one competing prefill to defer for one scheduler tick."""

from __future__ import annotations

from typing import Any

from waveslice.policy import WaveSlicePolicy
from waveslice.vllm.state import Phase12BeneficiarySignal, ScheduledRequestInfo


def phase12_scheduler_cashout_grade(
    *,
    policy: WaveSlicePolicy,
    candidate_id: str | None,
    selected_quality: float,
    value_signal: dict[str, float] | None = None,
):
    if not candidate_id:
        return None
    soft = max(0.0, min(1.0, policy.phase12_phase2_scheduler_cashout_soft_floor))
    hard = max(soft + 1e-6, min(1.0, policy.phase12_phase2_scheduler_cashout_quality_floor))
    quality = max(0.0, min(1.0, selected_quality))
    signal = value_signal or {"value_score": 1.0, "net_value": 1.0}
    value_score = max(0.0, min(1.0, float(signal.get("value_score", 0.0) or 0.0)))
    net_value = float(signal.get("net_value", value_score) or 0.0)
    fields = {
        name: float(signal.get(name, 0.0) or 0.0)
        for name in (
            "gain_score",
            "cost_score",
            "candidate_size_quality",
        )
    }
    base = {
        "allowed": False,
        "soft_floor": soft,
        "hard_floor": hard,
        "strength": 0.0,
        "value_score": value_score,
        "net_value": net_value,
        **fields,
    }
    if quality < soft or net_value <= 0:
        return base
    strength = 1.0 if quality >= hard else (quality - soft) / max(1e-6, hard - soft)
    return {**base, "allowed": True, "strength": max(0.0, min(1.0, strength * value_score))}


def phase12_scheduler_cashout_cooldown_for_grade(
    *, policy: WaveSlicePolicy, grade: dict[str, Any] | None
) -> int:
    maximum = max(0, policy.phase12_phase2_scheduler_cashout_cooldown_ticks)
    if not maximum or not grade:
        return 0
    score = 0.5 * max(0.0, min(1.0, grade.get("strength", 0.0))) + 0.5 * max(
        0.0, min(1.0, grade.get("value_score", 0.0))
    )
    return 0 if score >= 0.70 else min(1, maximum) if score >= 0.32 else maximum


def phase12_scheduler_cashout_value_signal(
    *,
    req_infos: list[ScheduledRequestInfo],
    beneficiary_signal: Phase12BeneficiarySignal,
    candidate_id: str | None,
) -> dict[str, float]:
    if not req_infos:
        return {"gain_score": 0.0, "cost_score": 1.0, "value_score": 0.0, "net_value": -1.0}
    prefills = [info for info in req_infos if info.remaining_tokens > 1]
    candidate = next((info for info in prefills if info.request_id == candidate_id), None)
    if candidate is None:
        return {"gain_score": 0.0, "cost_score": 1.0, "value_score": 0.0, "net_value": -1.0}
    costs = [max(1, info.expected_chunk_tokens) * max(1, info.lora_rank) for info in prefills]
    candidate_cost = max(1, candidate.expected_chunk_tokens) * max(1, candidate.lora_rank)
    candidate_size = candidate_cost / max(costs)
    gain = (
        0.70 * beneficiary_signal.beneficiary_selected_quality
        + 0.30 * beneficiary_signal.beneficiary_fraction
    )
    cost = 0.50 * candidate_size
    net = gain - cost
    return {
        "gain_score": gain,
        "cost_score": cost,
        "value_score": max(0.0, min(1.0, net)),
        "net_value": net,
        "candidate_size_quality": candidate_size,
    }


def phase12_cashout_candidate_id(
    *,
    req_infos: list[ScheduledRequestInfo],
    beneficiary_signal: Phase12BeneficiarySignal,
) -> str | None:
    selected = set(map(str, beneficiary_signal.beneficiary_selected_ids))
    candidates = [
        info for info in req_infos if info.remaining_tokens > 1 and info.request_id not in selected
    ]
    if not candidates:
        return None
    anchor_id = str(beneficiary_signal.long_anchor_id or "")
    if any(info.request_id == anchor_id for info in candidates):
        return anchor_id
    return max(
        candidates,
        key=lambda info: (
            info.remaining_tokens * max(1, info.lora_rank),
            info.remaining_tokens,
        ),
    ).request_id
