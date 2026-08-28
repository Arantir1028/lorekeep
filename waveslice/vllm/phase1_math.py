"""Phase-I pressure, budget, and chunk calculations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from waveslice.policy import WaveSlicePolicy
from waveslice.scheduling.slicer import WaveBaseSlicer
from waveslice.vllm.state import Phase1CohortStats


def need_wave_slice(lengths: list[int], policy: WaveSlicePolicy) -> bool:
    if len(lengths) < 2 or min(lengths) <= 0:
        return False
    short, long = min(lengths), max(lengths)
    extreme = (
        long >= policy.phase1_force_min_chunk and long >= short * policy.phase1_force_extreme_ratio
    )
    return extreme or (long > policy.min_long_seq and long >= short * policy.min_hetero_ratio)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _lerp(low: float, high: float, score: float) -> float:
    return low + (high - low) * _clamp(score)


def phase1_runtime_pressure_meta(
    *,
    policy: WaveSlicePolicy,
    cohort: Phase1CohortStats,
    queue_len: int,
    waiting_short_count: int,
    max_wait_us: float,
    virtual_cap_hit_rate: float,
    previous_wall_pressure: float = 0.0,
) -> dict[str, float]:
    queue = _clamp(max(0, queue_len) / max(1, policy.phase1_runtime_queue_high_watermark))
    waiting = _clamp(
        max(0, waiting_short_count) / max(1, policy.phase1_runtime_waiting_short_high_watermark)
    )
    wait = _clamp(max(0.0, max_wait_us) / max(1.0, policy.phase1_runtime_wait_us_high_watermark))
    long = _clamp(max(0, cohort.long_len) / max(1, policy.phase1_runtime_long_high_watermark))
    cap, mass = (
        _clamp(virtual_cap_hit_rate),
        _clamp(cohort.short_token_mass / max(1, cohort.long_len)),
    )
    urgency = _clamp(0.55 * waiting + 0.30 * wait + 0.15 * mass)
    wall = _clamp(0.35 * queue + 0.25 * long + 0.20 * cap + 0.20 * _clamp(previous_wall_pressure))
    effective = _clamp(wall * (1 - policy.phase1_runtime_urgency_discount * urgency))
    return {
        "queue_pressure": queue,
        "waiting_short_pressure": waiting,
        "wait_pressure": wait,
        "long_pressure": long,
        "cap_hit_pressure": cap,
        "short_urgency": urgency,
        "wall_pressure": wall,
        "effective_pressure": effective,
    }


def phase1_runtime_adapt_policy(policy: WaveSlicePolicy, meta: dict[str, float]):
    if not policy.phase1_runtime_adaptive_enabled:
        return policy, {}
    pressure = _clamp(meta.get("effective_pressure", 0.0))
    wall = _clamp(meta.get("wall_pressure", pressure))
    payload: dict[str, Any] = {
        "phase1_target_long_fraction": max(
            0.01,
            _lerp(
                policy.phase1_runtime_aggressive_long_fraction,
                policy.phase1_runtime_conservative_long_fraction,
                pressure,
            ),
        ),
        "phase1_ingress_target_chunk": max(
            1,
            round(
                _lerp(
                    policy.phase1_runtime_aggressive_ingress_target_chunk,
                    policy.phase1_runtime_conservative_ingress_target_chunk,
                    pressure,
                )
            ),
        ),
    }
    if policy.phase2_runtime_adaptive_enabled:
        for target, low, high in (
            (
                "phase2_min_hetero_ratio",
                "phase2_runtime_low_pressure_min_hetero_ratio",
                "phase2_runtime_high_pressure_min_hetero_ratio",
            ),
            (
                "phase2_min_pressure_ratio",
                "phase2_runtime_low_pressure_min_pressure_ratio",
                "phase2_runtime_high_pressure_min_pressure_ratio",
            ),
            (
                "phase2_min_long_prefill",
                "phase2_runtime_low_pressure_min_long_prefill",
                "phase2_runtime_high_pressure_min_long_prefill",
            ),
        ):
            value = _lerp(getattr(policy, low), getattr(policy, high), wall)
            payload[target] = value if "ratio" in target else round(value)
    adapted = replace(policy, **payload)
    detail = {key: float(value) for key, value in meta.items()}
    for name in (
        "phase1_target_long_fraction",
        "phase1_ingress_target_chunk",
        "phase2_min_hetero_ratio",
        "phase2_min_pressure_ratio",
        "phase2_min_long_prefill",
    ):
        detail[name] = float(getattr(adapted, name))
    return adapted, detail


def compute_budget(
    best_chunk: int,
    short_len: int,
    long_len: int,
    short_token_mass: int,
    queue_len: int,
    policy: WaveSlicePolicy,
    original_budget: Any,
    baseline_chunk: int | None = None,
) -> int | None:
    del long_len
    if not isinstance(original_budget, int) or original_budget <= 0:
        return None
    inflation = min(
        1024,
        short_len * policy.short_escape_multiplier
        + int(max(0, short_token_mass) * policy.phase1_budget_short_mass_factor)
        + max(0, queue_len) * max(0, policy.phase1_budget_queue_bonus),
    )
    candidate = best_chunk + inflation + max(0, policy.phase1_budget_bonus_tokens)
    if baseline_chunk and baseline_chunk > 0:
        candidate = min(
            candidate,
            max(best_chunk, baseline_chunk + inflation + max(0, policy.phase1_budget_bonus_tokens)),
        )
    return max(1, min(candidate, policy.max_budget_cap))


def compute_explicit_plan_budget(
    *,
    best_chunk: int,
    short_len: int,
    short_token_mass: int,
    policy: WaveSlicePolicy,
    original_budget: Any,
    baseline_chunk: int | None,
) -> int | None:
    if not isinstance(original_budget, int) or original_budget <= 0:
        return None
    inflation = min(
        max(0, policy.phase1_explicit_budget_cap_tokens),
        max(short_len, int(max(0, short_token_mass) * policy.phase1_budget_short_mass_factor)),
    )
    ceilings = [best_chunk + inflation, original_budget, policy.max_budget_cap]
    if baseline_chunk and baseline_chunk > 0:
        ceilings.append(baseline_chunk)
    return max(best_chunk, min(ceilings))


def phase1_authoritative_short_floor(
    policy: WaveSlicePolicy, *, short_len: int, target: int
) -> int:
    short = max(1, int(short_len))
    return max(1, min(short, target)) if policy.phase1_ingress_exact_chunk else short


def phase1_effective_ingress_min_chunk(
    policy: WaveSlicePolicy, *, target: int | None = None
) -> int:
    minimum = max(1, policy.phase1_ingress_min_chunk)
    return min(minimum, max(1, target)) if policy.phase1_ingress_exact_chunk and target else minimum


def phase1_effective_ingress_target_chunk(policy: WaveSlicePolicy, *, target: int) -> int:
    target = max(1, int(target))
    return (
        min(target, max(1, policy.phase1_ingress_target_chunk))
        if policy.phase1_ingress_exact_chunk
        else target
    )


def phase1_authoritative_chunk(
    policy: WaveSlicePolicy,
    slicer: WaveBaseSlicer,
    *,
    target: int,
    short_len: int = 0,
    upper: int | None = None,
) -> int:
    target = phase1_effective_ingress_target_chunk(policy, target=target)
    minimum = phase1_effective_ingress_min_chunk(policy, target=target)
    target = max(minimum, min(target, max(minimum, policy.phase1_ingress_max_chunk)))
    floor = phase1_authoritative_short_floor(policy, short_len=short_len, target=target)
    upper = max(floor + 1, target if upper is None else int(upper))
    target = min(target, upper)
    chosen = target if policy.phase1_ingress_exact_chunk else slicer._conservative_map_down(target)
    return max(floor, min(chosen, upper))


def phase1_baseline_chunk_proxy(
    *,
    long_len: int,
    original_budget: Any,
    original_threshold: Any,
    scheduler_cfg: Any,
    policy: WaveSlicePolicy,
) -> int | None:
    if not policy.enable_phase1_baseline_relative or not getattr(
        scheduler_cfg, "enable_chunked_prefill", True
    ):
        return None
    candidates = [long_len] + [
        value
        for value in (original_budget, original_threshold)
        if isinstance(value, int) and value > 0
    ]
    baseline = min(candidates)
    return max(1, baseline) if baseline < long_len else None


def phase1_adjusted_queue_len(
    cohort: Phase1CohortStats, queue_len: int, policy: WaveSlicePolicy
) -> int:
    extra = max(0, cohort.short_count - 1) * max(0, policy.phase1_cohort_queue_bonus)
    units = cohort.short_token_mass / max(1, cohort.representative_short_len)
    extra += int(max(0.0, units - 1) * max(0.0, policy.phase1_cohort_mass_queue_factor))
    return max(1, queue_len + extra)


def phase1_cohort_target_len(cohort: Phase1CohortStats, policy: WaveSlicePolicy) -> int:
    mean = max(1, round(cohort.short_token_mass / max(1, cohort.short_count)))
    short = max(policy.phase1_force_min_chunk, mean * policy.phase1_target_short_mul)
    mass = max(policy.phase1_force_min_chunk, mean * policy.phase1_cohort_target_mass_factor)
    fraction = max(
        policy.phase1_force_min_chunk, cohort.long_len * policy.phase1_target_long_fraction
    )
    return max(1, int(min(fraction, max(short, mass), cohort.long_len - 1)))


def phase1_effective_short_token_mass(
    lengths: list[int], *, short_len: int, best_chunk: int, policy: WaveSlicePolicy
) -> int:
    limit = max(best_chunk, int(short_len * max(1.0, policy.phase1_target_short_mul)))
    return max(short_len, sum(value for value in map(int, lengths) if 0 < value <= limit))


def maybe_force_phase1_chunk(
    *,
    cohort: Phase1CohortStats,
    queue_len: int,
    chosen_chunk: int,
    slicer: WaveBaseSlicer,
    policy: WaveSlicePolicy,
) -> int:
    short, long = (
        max(1, cohort.representative_short_len),
        max(cohort.representative_short_len + 1, cohort.long_len),
    )
    chosen = max(short, min(chosen_chunk, long))
    force = (
        long / short >= policy.phase1_force_extreme_ratio
        and queue_len >= policy.phase1_force_queue_len
        and long >= policy.phase1_force_min_chunk
    )
    if not force:
        return chosen
    forced = slicer._conservative_map_down(
        max(short + 1, min(phase1_cohort_target_len(cohort, policy), long - 1))
    )
    forced = max(short, min(forced, long - 1))
    return forced if forced < chosen or chosen >= long else chosen
