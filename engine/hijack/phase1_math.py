from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from engine.base_slicer import WaveBaseSlicer
from engine.hijack.runtime_state import WaveSlicePolicy
from engine.hijack.types import _Phase1CohortStats


def need_wave_slice(lengths: list[int], policy: WaveSlicePolicy) -> bool:
    if len(lengths) < 2:
        return False
    s_min = min(lengths)
    s_max = max(lengths)
    if s_min <= 0:
        return False
    if (
        s_max >= int(max(1, policy.phase1_force_min_chunk))
        and s_max >= int(s_min * max(1.0, policy.phase1_force_extreme_ratio))
    ):
        return True
    if s_max <= policy.min_long_seq:
        return False
    return s_max >= int(s_min * policy.min_hetero_ratio)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lerp(low: float, high: float, score: float) -> float:
    score = _clamp01(score)
    return float(low) + (float(high) - float(low)) * score


def phase1_runtime_pressure_meta(
    *,
    policy: WaveSlicePolicy,
    cohort: _Phase1CohortStats,
    queue_len: int,
    waiting_short_count: int,
    max_wait_us: float,
    virtual_cap_hit_rate: float,
    previous_wall_pressure: float = 0.0,
) -> dict[str, float]:
    queue_pressure = _clamp01(
        float(max(0, int(queue_len)))
        / float(max(1, int(policy.phase1_runtime_queue_high_watermark)))
    )
    waiting_short_pressure = _clamp01(
        float(max(0, int(waiting_short_count)))
        / float(max(1, int(policy.phase1_runtime_waiting_short_high_watermark)))
    )
    wait_pressure = _clamp01(
        float(max(0.0, max_wait_us))
        / float(max(1.0, float(policy.phase1_runtime_wait_us_high_watermark)))
    )
    long_pressure = _clamp01(
        float(max(0, int(cohort.long_len)))
        / float(max(1, int(policy.phase1_runtime_long_high_watermark)))
    )
    cap_hit_pressure = _clamp01(virtual_cap_hit_rate)
    short_mass_ratio = _clamp01(
        float(max(0, int(cohort.short_token_mass)))
        / float(max(1, int(cohort.long_len)))
    )
    urgency = _clamp01(
        (0.55 * waiting_short_pressure)
        + (0.30 * wait_pressure)
        + (0.15 * short_mass_ratio)
    )
    raw_wall_pressure = _clamp01(
        (0.35 * queue_pressure)
        + (0.25 * long_pressure)
        + (0.20 * cap_hit_pressure)
        + (0.20 * _clamp01(previous_wall_pressure))
    )
    effective_pressure = _clamp01(
        raw_wall_pressure
        * (1.0 - (float(policy.phase1_runtime_urgency_discount) * urgency))
    )
    return {
        "queue_pressure": queue_pressure,
        "waiting_short_pressure": waiting_short_pressure,
        "wait_pressure": wait_pressure,
        "long_pressure": long_pressure,
        "cap_hit_pressure": cap_hit_pressure,
        "short_urgency": urgency,
        "wall_pressure": raw_wall_pressure,
        "effective_pressure": effective_pressure,
    }


def phase1_runtime_adapt_policy(
    policy: WaveSlicePolicy,
    meta: dict[str, float],
) -> tuple[WaveSlicePolicy, dict[str, float]]:
    if not bool(policy.phase1_runtime_adaptive_enabled):
        return policy, {}
    pressure = _clamp01(meta.get("effective_pressure", 0.0))
    wall_pressure = _clamp01(meta.get("wall_pressure", pressure))

    target_fraction = _lerp(
        float(policy.phase1_runtime_aggressive_long_fraction),
        float(policy.phase1_runtime_conservative_long_fraction),
        pressure,
    )
    ingress_target_chunk = int(
        round(
            _lerp(
                float(policy.phase1_runtime_aggressive_ingress_target_chunk),
                float(policy.phase1_runtime_conservative_ingress_target_chunk),
                pressure,
            )
        )
    )
    payload: dict[str, Any] = {
        "phase1_target_long_fraction": max(0.01, float(target_fraction)),
        "phase1_ingress_target_chunk": max(1, int(ingress_target_chunk)),
    }

    if bool(policy.phase2_runtime_adaptive_enabled):
        phase2_pressure = wall_pressure
        payload.update(
            {
                "phase2_min_hetero_ratio": _lerp(
                    float(policy.phase2_runtime_low_pressure_min_hetero_ratio),
                    float(policy.phase2_runtime_high_pressure_min_hetero_ratio),
                    phase2_pressure,
                ),
                "phase2_min_pressure_ratio": _lerp(
                    float(policy.phase2_runtime_low_pressure_min_pressure_ratio),
                    float(policy.phase2_runtime_high_pressure_min_pressure_ratio),
                    phase2_pressure,
                ),
                "phase2_min_long_prefill": int(
                    round(
                        _lerp(
                            float(policy.phase2_runtime_low_pressure_min_long_prefill),
                            float(policy.phase2_runtime_high_pressure_min_long_prefill),
                            phase2_pressure,
                        )
                    )
                ),
                "phase2_execution_escape_spillover_cap": int(
                    round(
                        _lerp(
                            float(policy.phase2_runtime_low_pressure_escape_spillover_cap),
                            float(policy.phase2_runtime_high_pressure_escape_spillover_cap),
                            phase2_pressure,
                        )
                    )
                ),
                "phase2_execution_escape_max_active": int(
                    round(
                        _lerp(
                            float(policy.phase2_runtime_low_pressure_escape_max_active),
                            float(policy.phase2_runtime_high_pressure_escape_max_active),
                            phase2_pressure,
                        )
                    )
                ),
            }
        )
        disable_below = float(policy.phase2_runtime_disable_execution_escape_below_pressure)
        if disable_below >= 0.0 and phase2_pressure < disable_below:
            payload["phase2_enable_execution_escape"] = False

    adapted = replace(policy, **payload)
    return adapted, {
        **{key: float(value) for key, value in meta.items()},
        "phase1_target_long_fraction": float(payload["phase1_target_long_fraction"]),
        "phase1_ingress_target_chunk": float(payload["phase1_ingress_target_chunk"]),
        "phase2_min_hetero_ratio": float(getattr(adapted, "phase2_min_hetero_ratio", 0.0)),
        "phase2_min_pressure_ratio": float(getattr(adapted, "phase2_min_pressure_ratio", 0.0)),
        "phase2_min_long_prefill": float(getattr(adapted, "phase2_min_long_prefill", 0.0)),
        "phase2_execution_escape_spillover_cap": float(getattr(adapted, "phase2_execution_escape_spillover_cap", 0.0)),
        "phase2_execution_escape_max_active": float(getattr(adapted, "phase2_execution_escape_max_active", 0.0)),
        "phase2_enable_execution_escape": (
            1.0 if bool(getattr(adapted, "phase2_enable_execution_escape", False)) else 0.0
        ),
    }


def compute_budget(
    best_chunk: int,
    short_len: int,
    long_len: int,
    short_token_mass: int,
    queue_len: int,
    policy: WaveSlicePolicy,
    original_budget: Any,
    baseline_chunk: Optional[int] = None,
) -> Optional[int]:
    _ = long_len
    if not isinstance(original_budget, int) or original_budget <= 0:
        return None
    escape_allowance = short_len * policy.short_escape_multiplier
    mass_allowance = int(
        max(0.0, float(short_token_mass))
        * max(0.0, float(policy.phase1_budget_short_mass_factor))
    )
    queue_allowance = int(max(0, queue_len)) * int(max(0, policy.phase1_budget_queue_bonus))
    max_inflation = 1024
    total_inflation = min(max_inflation, escape_allowance + mass_allowance + queue_allowance)

    candidate = best_chunk + total_inflation + int(max(0, policy.phase1_budget_bonus_tokens))
    candidate = max(best_chunk, candidate)
    if baseline_chunk is not None and int(baseline_chunk) > 0:
        baseline_chunk = max(1, int(baseline_chunk))
        baseline_ceiling = max(
            best_chunk,
            baseline_chunk + total_inflation + int(max(0, policy.phase1_budget_bonus_tokens)),
        )
        candidate = min(candidate, baseline_ceiling)
    candidate = min(candidate, policy.max_budget_cap)
    return max(1, candidate)


def compute_explicit_plan_budget(
    *,
    best_chunk: int,
    short_len: int,
    short_token_mass: int,
    policy: WaveSlicePolicy,
    original_budget: Any,
    baseline_chunk: Optional[int],
) -> Optional[int]:
    if not isinstance(original_budget, int) or original_budget <= 0:
        return None
    explicit_inflation = min(
        int(max(0, policy.phase1_explicit_budget_cap_tokens)),
        max(
            short_len,
            int(max(0.0, float(short_token_mass)) * max(0.0, float(policy.phase1_budget_short_mass_factor))),
        ),
    )
    candidate = max(1, int(best_chunk) + explicit_inflation)
    if baseline_chunk is not None and int(baseline_chunk) > 0:
        candidate = min(candidate, int(baseline_chunk))
    candidate = min(candidate, int(original_budget), int(policy.max_budget_cap))
    return max(int(best_chunk), candidate)


def phase1_authoritative_short_floor(
    policy: WaveSlicePolicy,
    *,
    short_len: int,
    target: int,
) -> int:
    base_floor = max(1, int(short_len))
    if bool(policy.phase1_ingress_exact_chunk):
        return max(1, min(base_floor, int(target)))
    return base_floor


def phase1_effective_ingress_min_chunk(
    policy: WaveSlicePolicy,
    *,
    target: Optional[int] = None,
) -> int:
    ingress_min = max(1, int(policy.phase1_ingress_min_chunk))
    if bool(policy.phase1_ingress_exact_chunk) and target is not None and int(target) > 0:
        ingress_min = min(ingress_min, max(1, int(target)))
    return ingress_min


def phase1_effective_ingress_target_chunk(
    policy: WaveSlicePolicy,
    *,
    target: int,
) -> int:
    effective_target = max(1, int(target))
    if bool(policy.phase1_ingress_exact_chunk):
        effective_target = min(
            effective_target,
            max(1, int(policy.phase1_ingress_target_chunk)),
        )
    return effective_target


def phase1_authoritative_chunk(
    policy: WaveSlicePolicy,
    slicer: WaveBaseSlicer,
    *,
    target: int,
    short_len: int = 0,
    upper: Optional[int] = None,
) -> int:
    target = phase1_effective_ingress_target_chunk(policy, target=int(target))
    ingress_min = phase1_effective_ingress_min_chunk(policy, target=int(target))
    ingress_max = max(ingress_min, int(policy.phase1_ingress_max_chunk))
    target = max(ingress_min, min(int(target), ingress_max))
    floor = phase1_authoritative_short_floor(policy, short_len=int(short_len), target=int(target))
    if upper is None:
        upper = max(floor + 1, target)
    upper = max(floor + 1, int(upper))
    target = min(target, upper)
    if bool(policy.phase1_ingress_exact_chunk):
        return max(floor, min(int(target), upper))
    chunk = int(slicer._conservative_map_down(int(target)))
    return max(floor, min(chunk, upper))


def phase1_baseline_chunk_proxy(
    *,
    long_len: int,
    original_budget: Any,
    original_threshold: Any,
    scheduler_cfg: Any,
    policy: WaveSlicePolicy,
) -> Optional[int]:
    if not bool(policy.enable_phase1_baseline_relative):
        return None
    chunked_enabled = True
    try:
        chunked_enabled = bool(getattr(scheduler_cfg, "enable_chunked_prefill", True))
    except Exception:
        chunked_enabled = True
    if not chunked_enabled:
        return None

    candidates = [max(1, int(long_len))]
    if isinstance(original_budget, int) and original_budget > 0:
        candidates.append(int(original_budget))
    if isinstance(original_threshold, int) and original_threshold > 0:
        candidates.append(int(original_threshold))
    baseline_chunk = min(candidates)
    if baseline_chunk >= int(long_len):
        return None
    return max(1, baseline_chunk)


def phase1_adjusted_queue_len(
    cohort: _Phase1CohortStats,
    queue_len: int,
    policy: WaveSlicePolicy,
) -> int:
    extra = max(0, int(cohort.short_count) - 1) * int(max(0, policy.phase1_cohort_queue_bonus))
    mass_units = float(cohort.short_token_mass) / float(max(1, cohort.representative_short_len))
    extra += int(max(0.0, mass_units - 1.0) * max(0.0, float(policy.phase1_cohort_mass_queue_factor)))
    return max(1, int(queue_len) + extra)


def phase1_cohort_target_len(
    cohort: _Phase1CohortStats,
    policy: WaveSlicePolicy,
) -> int:
    mean_short = max(1, int(round(cohort.short_token_mass / max(1, cohort.short_count))))
    target_by_short = int(max(policy.phase1_force_min_chunk, mean_short * float(policy.phase1_target_short_mul)))
    target_by_mass = int(max(policy.phase1_force_min_chunk, mean_short * float(policy.phase1_cohort_target_mass_factor)))
    target_by_fraction = int(max(policy.phase1_force_min_chunk, cohort.long_len * float(policy.phase1_target_long_fraction)))
    return max(1, min(target_by_fraction, max(target_by_short, target_by_mass), cohort.long_len - 1))


def phase1_effective_short_token_mass(
    lengths: list[int],
    *,
    short_len: int,
    best_chunk: int,
    policy: WaveSlicePolicy,
) -> int:
    limit = max(int(best_chunk), int(short_len * max(1.0, policy.phase1_target_short_mul)))
    mass = 0
    for val in lengths:
        iv = int(val)
        if iv <= 0:
            continue
        if iv <= limit:
            mass += iv
    return max(short_len, mass)


def maybe_force_phase1_chunk(
    *,
    cohort: _Phase1CohortStats,
    queue_len: int,
    chosen_chunk: int,
    slicer: WaveBaseSlicer,
    policy: WaveSlicePolicy,
) -> tuple[int, bool]:
    short_len = max(1, int(cohort.representative_short_len))
    long_len = max(short_len + 1, int(cohort.long_len))
    chosen_chunk = max(short_len, min(int(chosen_chunk), long_len))
    ratio = float(long_len) / float(max(1, short_len))
    should_force = (
        ratio >= float(policy.phase1_force_extreme_ratio)
        and queue_len >= int(policy.phase1_force_queue_len)
        and long_len >= int(policy.phase1_force_min_chunk)
    )
    if not should_force:
        return chosen_chunk, False

    cohort_target = phase1_cohort_target_len(cohort, policy)
    forced_cap = max(short_len + 1, min(int(cohort_target), long_len - 1))
    forced_chunk = slicer._conservative_map_down(forced_cap)
    forced_chunk = max(short_len, min(int(forced_chunk), long_len - 1))
    if forced_chunk < chosen_chunk or chosen_chunk >= long_len:
        return forced_chunk, True
    return chosen_chunk, False
