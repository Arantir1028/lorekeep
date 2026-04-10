from __future__ import annotations

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
