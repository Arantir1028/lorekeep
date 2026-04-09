from __future__ import annotations

from typing import Any, Callable

from engine.hijack.runtime_state import WaveSlicePolicy
from engine.hijack.types import _Phase2Decision

SelectiveGate = Callable[[list[int], list[int], WaveSlicePolicy, bool], tuple[bool, float, float, bool]]
MixedEscapeOk = Callable[[list[int], int, float, float, bool, WaveSlicePolicy, bool], bool]
PressureRatio = Callable[[list[int], list[int]], float]


def phase2_decide_from_prefill_window(
    *,
    policy: WaveSlicePolicy,
    prefill_lens: list[int],
    num_prefills: int,
    num_decode_tokens: int,
    lora_ranks: list[int],
    strict_mode: bool,
    phase12_ready: bool,
    phase12_reason: str,
    reason_suffix: str = "",
    selective_gate: SelectiveGate,
    mixed_escape_ok_fn: MixedEscapeOk,
    pressure_ratio_fn: PressureRatio,
) -> _Phase2Decision:
    selective_ok, ratio, pressure_ratio, lora_rank_hetero = selective_gate(
        prefill_lens,
        lora_ranks,
        policy,
        strict_mode,
    )
    suffix = str(reason_suffix or "")
    if not phase12_ready:
        return _Phase2Decision(
            False,
            f"{phase12_reason}{suffix}",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    if strict_mode and num_decode_tokens > 0:
        return _Phase2Decision(
            False,
            "strict_no_mixed_prefill_decode",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    if (
        not strict_mode
        and policy.phase2_enable_mixed_prefill_decode
        and num_prefills > 0
        and num_decode_tokens > 0
    ):
        mixed_escape_ok = mixed_escape_ok_fn(
            prefill_lens,
            num_decode_tokens,
            ratio,
            pressure_ratio,
            lora_rank_hetero,
            policy,
            strict_mode,
        )
        if bool(policy.phase2_selective_only) and not selective_ok:
            if mixed_escape_ok:
                return _Phase2Decision(
                    True,
                    f"selective_mixed_prefill_decode_soft{suffix}",
                    prefill_lens,
                    num_prefills,
                    num_decode_tokens,
                    lora_ranks,
                )
            return _Phase2Decision(
                False,
                f"mixed_prefill_decode_not_extreme{suffix}",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )
        return _Phase2Decision(
            True,
            f"selective_mixed_prefill_decode{suffix}",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    if not strict_mode and num_prefills >= max(1, policy.phase2_min_prefill_count):
        if bool(policy.phase2_selective_only) and not selective_ok:
            return _Phase2Decision(
                False,
                f"prefill_batch_not_extreme{suffix}",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )
        return _Phase2Decision(
            True,
            f"selective_lora_extreme_prefill{suffix}",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    min_prefill_count = max(policy.phase2_min_prefill_count, 2 if strict_mode else 1)
    if len(prefill_lens) < min_prefill_count:
        return _Phase2Decision(
            False,
            "insufficient_prefill_batch",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )

    min_len = max(1, min(prefill_lens))
    max_len = max(prefill_lens)
    ratio = float(max_len) / float(min_len)
    pressure_ratio = pressure_ratio_fn(prefill_lens, lora_ranks)
    min_hetero_ratio = max(policy.phase2_min_hetero_ratio, 3.0 if strict_mode else 0.0)
    min_long_prefill = max(policy.phase2_min_long_prefill, 512 if strict_mode else 0)
    if ratio < min_hetero_ratio:
        if lora_rank_hetero and pressure_ratio >= float(policy.phase2_min_pressure_ratio) and selective_ok:
            return _Phase2Decision(
                True,
                ("strict_" if strict_mode else "") + f"selective_lora_rank_pressure_prefill{suffix}",
                prefill_lens,
                num_prefills,
                num_decode_tokens,
                lora_ranks,
            )
        return _Phase2Decision(
            False,
            "strict_hetero_ratio_too_low" if strict_mode else "hetero_ratio_too_low",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )
    if max_len < min_long_prefill:
        return _Phase2Decision(
            False,
            "strict_long_prefill_too_short" if strict_mode else "long_prefill_too_short",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )
    if bool(policy.phase2_selective_only) and not selective_ok:
        return _Phase2Decision(
            False,
            ("strict_" if strict_mode else "") + f"prefill_not_extreme{suffix}",
            prefill_lens,
            num_prefills,
            num_decode_tokens,
            lora_ranks,
        )
    return _Phase2Decision(
        True,
        ("strict_" if strict_mode else "") + f"selective_lora_extreme_prefill{suffix}",
        prefill_lens,
        num_prefills,
        num_decode_tokens,
        lora_ranks,
    )
