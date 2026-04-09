from __future__ import annotations

from engine.hijack.runtime_state import WaveSlicePolicy


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
    return rank_ratio >= float(policy.phase2_min_rank_ratio) and rank_gap >= int(policy.phase2_min_rank_gap)


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
    *,
    prefill_lens: list[int],
    lora_ranks: list[int],
    policy: WaveSlicePolicy,
    strict_mode: bool,
) -> tuple[bool, float, float, bool]:
    pos_prefills = [int(v) for v in prefill_lens if int(v) > 0]
    if not pos_prefills:
        return False, 1.0, 1.0, False
    min_len = max(1, min(pos_prefills))
    max_len = max(pos_prefills)
    length_ratio = float(max_len) / float(min_len)
    pressure_ratio = phase2_pressure_ratio(pos_prefills, lora_ranks)
    rank_ratio = phase2_rank_ratio(lora_ranks)
    ratio = max(length_ratio, rank_ratio)
    lora_rank_hetero = phase2_has_lora_heterogeneity(lora_ranks, policy)
    need_rank_hetero = bool(policy.phase2_require_rank_hetero)
    if need_rank_hetero and not lora_rank_hetero:
        return False, ratio, pressure_ratio, lora_rank_hetero

    min_ratio = max(
        float(policy.phase2_min_hetero_ratio),
        3.0 if strict_mode else 0.0,
    )
    min_long_prefill = max(
        int(policy.phase2_min_long_prefill),
        512 if strict_mode else 0,
    )
    min_pressure = float(policy.phase2_min_pressure_ratio)
    selective = ratio >= min_ratio and max_len >= min_long_prefill and pressure_ratio >= min_pressure
    return selective, ratio, pressure_ratio, lora_rank_hetero


def phase2_mixed_escape_ok(
    *,
    prefill_lens: list[int],
    num_decode_tokens: int,
    ratio: float,
    pressure_ratio: float,
    lora_rank_hetero: bool,
    policy: WaveSlicePolicy,
    strict_mode: bool,
) -> bool:
    if strict_mode or num_decode_tokens <= 0:
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
