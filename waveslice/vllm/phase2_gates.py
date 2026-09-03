from __future__ import annotations

from waveslice.policy import WaveSlicePolicy
from waveslice.vllm.state import Phase12BeneficiarySignal, RuntimeState


def phase2_has_lora_heterogeneity(ranks: list[int], policy: WaveSlicePolicy) -> bool:
    if not policy.phase2_lora_rank_aware:
        return False
    pos = [int(r) for r in ranks if int(r) > 0]
    if len(pos) < max(2, int(policy.phase2_min_lora_count)):
        return False
    min_rank = max(1, min(pos))
    max_rank = max(pos)
    rank_ratio = float(max_rank) / float(min_rank)
    rank_gap = int(max_rank) - int(min_rank)
    return rank_ratio >= float(policy.phase2_min_rank_ratio) and rank_gap >= int(
        policy.phase2_min_rank_gap
    )


def phase2_rank_ratio(lora_ranks: list[int]) -> float:
    pos = [int(r) for r in lora_ranks if int(r) > 0]
    if len(pos) < 2:
        return 1.0
    min_rank = max(1, min(pos))
    max_rank = max(pos)
    return float(max_rank) / float(min_rank)


def phase2_pressure_ratio(prefill_lens: list[int], lora_ranks: list[int]) -> float:
    if not prefill_lens:
        return 1.0
    min_len = max(1, min(int(v) for v in prefill_lens if int(v) > 0))
    max_len = max(int(v) for v in prefill_lens if int(v) > 0)
    length_ratio = float(max_len) / float(min_len)
    pos_ranks = [int(r) for r in lora_ranks if int(r) > 0]
    if not pos_ranks:
        return length_ratio
    min_rank = max(1, min(pos_ranks))
    max_rank = max(pos_ranks)
    rank_ratio = float(max_rank) / float(min_rank)
    return length_ratio * rank_ratio


def phase2_selective_gate(
    *, prefill_lens: list[int], lora_ranks: list[int], policy: WaveSlicePolicy
) -> tuple[bool, float, float, bool]:
    pos_prefills = [int(v) for v in prefill_lens if int(v) > 0]
    if not pos_prefills:
        return (False, 1.0, 1.0, False)
    min_len = max(1, min(pos_prefills))
    max_len = max(pos_prefills)
    length_ratio = float(max_len) / float(min_len)
    pressure_ratio = phase2_pressure_ratio(pos_prefills, lora_ranks)
    rank_ratio = phase2_rank_ratio(lora_ranks)
    ratio = max(length_ratio, rank_ratio)
    lora_rank_hetero = phase2_has_lora_heterogeneity(lora_ranks, policy)
    need_rank_hetero = bool(policy.phase2_require_rank_hetero)
    if need_rank_hetero and (not lora_rank_hetero):
        return (False, ratio, pressure_ratio, lora_rank_hetero)
    min_ratio = float(policy.phase2_min_hetero_ratio)
    min_long_prefill = int(policy.phase2_min_long_prefill)
    min_pressure = float(policy.phase2_min_pressure_ratio)
    selective = (
        ratio >= min_ratio and max_len >= min_long_prefill and (pressure_ratio >= min_pressure)
    )
    return (selective, ratio, pressure_ratio, lora_rank_hetero)


def phase2_mixed_priority_ok(
    *,
    prefill_lens: list[int],
    num_decode_tokens: int,
    ratio: float,
    pressure_ratio: float,
    lora_rank_hetero: bool,
    policy: WaveSlicePolicy,
) -> bool:
    if num_decode_tokens <= 0:
        return False
    pos_prefills = [int(v) for v in prefill_lens if int(v) > 0]
    if not pos_prefills:
        return False
    max_len = max(pos_prefills)
    if max_len < int(policy.phase2_min_long_prefill):
        return False
    soft_ratio = max(1.25, float(policy.phase2_min_hetero_ratio) * 0.75)
    soft_pressure = max(1.5, float(policy.phase2_min_pressure_ratio))
    return ratio >= soft_ratio and (pressure_ratio >= soft_pressure or lora_rank_hetero)


def _soft_gate_decision(
    *,
    policy: WaveSlicePolicy,
    prefills: list[int],
    num_decode_tokens: int,
    lora_ranks: list[int],
    signal: Phase12BeneficiarySignal,
    cap_live: bool,
    recent_ttl: int,
) -> tuple[bool, str]:
    selective, ratio, pressure, rank_hetero = phase2_selective_gate(
        prefill_lens=prefills,
        lora_ranks=lora_ranks,
        policy=policy,
    )
    if not selective:
        return (False, "joint_soft_not_selective")
    if not signal.beneficiary_selected_ids:
        return (False, "joint_soft_no_beneficiary")
    if cap_live or recent_ttl > 0:
        return (True, "joint_soft_recent_phase1")
    if not policy.phase12_phase2_requires_recent_phase1:
        return (True, "joint_soft_recent_phase1_not_required")
    strong_prefill = (
        max(prefills)
        >= max(policy.phase2_min_long_prefill, policy.phase12_phase2_soft_min_long_prefill)
        and signal.beneficiary_selected_quality >= policy.phase12_phase2_beneficiary_quality_floor
    )
    if strong_prefill:
        return (True, "joint_soft_strong_prefill")
    mixed = policy.phase12_phase2_soft_allow_mixed_decode and phase2_mixed_priority_ok(
        prefill_lens=prefills,
        num_decode_tokens=num_decode_tokens,
        ratio=ratio,
        pressure_ratio=pressure,
        lora_rank_hetero=rank_hetero,
        policy=policy,
    )
    if mixed:
        return (True, "joint_soft_mixed_priority")
    return (False, "joint_soft_waiting_for_phase1")


def phase12_joint_phase2_ready(
    *,
    state: RuntimeState,
    policy: WaveSlicePolicy,
    prefill_lens: list[int],
    num_decode_tokens: int,
    lora_ranks: list[int],
    signal: Phase12BeneficiarySignal,
) -> tuple[bool, str]:
    if not (policy.enable_phase1_scheduler and policy.enable_phase2_scheduler):
        return (True, "joint_disabled")
    if not policy.phase12_joint_coordination:
        return (True, "joint_coordination_off")
    cap_live = bool(state.phase1_virtual_token_caps)
    recent_ttl = int(state.phase12_recent_phase1_apply_ttl)
    mode = policy.phase12_phase2_gate_mode.strip().lower()
    if mode == "hard":
        if cap_live:
            return (True, "joint_recent_phase1_cap_live")
        if recent_ttl > 0:
            return (True, "joint_recent_phase1_ttl")
        if policy.phase12_phase2_requires_recent_phase1:
            return (False, "joint_waiting_for_phase1")
        return (True, "joint_recent_phase1_not_required")
    if mode != "soft":
        raise ValueError(f"unknown Phase-II gate mode: {policy.phase12_phase2_gate_mode}")
    prefills = [int(value) for value in prefill_lens if int(value) > 0]
    if not prefills:
        return (False, "joint_soft_no_prefill")
    return _soft_gate_decision(
        policy=policy,
        prefills=prefills,
        num_decode_tokens=num_decode_tokens,
        lora_ranks=lora_ranks,
        signal=signal,
        cap_live=cap_live,
        recent_ttl=recent_ttl,
    )
