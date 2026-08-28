"""Algorithm policy knobs for WaveSlice Phase I and Phase II."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaveSlicePolicy:
    """Runtime knobs for Phase I/II behavior and metrics."""

    # Phase I (scheduler)
    enable_phase1_scheduler: bool = True
    min_hetero_ratio: float = 3.0
    min_long_seq: int = 384
    short_escape_multiplier: int = 12
    max_budget_cap: int = 8192
    enable_sjf_reorder: bool = True
    queue_reorder_mode: str = "sjf"  # sjf | hrrn | aging
    queue_reorder_aging_quantum_us: float = 20_000.0
    allow_phase1_with_lora: bool = False
    allow_phase1_threshold_with_lora: bool = True
    allow_phase1_budget_with_lora: bool = False
    enable_phase1_dynamic_threshold: bool = True
    enable_phase1_budget_guidance: bool = True
    enable_phase1_baseline_relative: bool = True
    enable_phase1_explicit_plan: bool = True
    enable_phase1_direct_explicit_override: bool = True
    phase1_ingress_direct_authoritative: bool = True
    scheduler_objective_mode: str = "fair_escape"  # fair_escape | pure_gain
    phase1_force_extreme_ratio: float = 6.0
    phase1_force_queue_len: int = 1
    phase1_force_min_chunk: int = 128
    phase1_ingress_exact_chunk: bool = True
    phase1_ingress_target_chunk: int = 384
    phase1_ingress_min_chunk: int = 256
    phase1_ingress_max_chunk: int = 512
    phase1_target_short_mul: float = 4.0
    phase1_target_long_fraction: float = 0.33
    phase1_budget_short_mass_factor: float = 1.75
    phase1_budget_bonus_tokens: int = 256
    phase1_budget_queue_bonus: int = 64
    phase1_explicit_budget_cap_tokens: int = 512
    phase1_cohort_queue_bonus: int = 2
    phase1_cohort_mass_queue_factor: float = 0.5
    phase1_cohort_target_mass_factor: float = 1.0
    phase1_runtime_adaptive_enabled: bool = False
    phase1_runtime_aggressive_long_fraction: float = 0.33
    phase1_runtime_conservative_long_fraction: float = 0.50
    phase1_runtime_aggressive_ingress_target_chunk: int = 768
    phase1_runtime_conservative_ingress_target_chunk: int = 1536
    phase1_runtime_queue_high_watermark: int = 8
    phase1_runtime_waiting_short_high_watermark: int = 4
    phase1_runtime_wait_us_high_watermark: float = 1_000_000.0
    phase1_runtime_long_high_watermark: int = 3072
    phase1_runtime_urgency_discount: float = 0.55
    phase1_runtime_ema_alpha: float = 0.35
    # Phase II (scheduler-bound priority cashout)
    enable_phase2_scheduler: bool = False
    phase2_min_prefill_count: int = 1
    phase2_min_hetero_ratio: float = 2.0
    phase2_min_long_prefill: int = 256
    phase2_enable_scheduler_cashout: bool = False
    phase2_lora_rank_aware: bool = True
    phase2_min_lora_count: int = 2
    phase2_min_rank_ratio: float = 1.5
    phase2_min_rank_gap: int = 4
    phase2_min_pressure_ratio: float = 2.0
    phase2_require_rank_hetero: bool = False
    phase12_joint_coordination: bool = True
    phase12_joint_min_chunk: int = 512
    phase12_phase2_requires_recent_phase1: bool = True
    phase12_phase2_recent_ttl: int = 4
    phase12_phase2_gate_mode: str = "soft"  # hard | soft
    phase12_phase2_soft_min_long_prefill: int = 512
    phase12_phase2_soft_allow_mixed_decode: bool = True
    phase12_phase2_beneficiary_prefill_scale: float = 1.5
    phase12_phase2_beneficiary_score_threshold: float = 0.55
    phase12_phase2_beneficiary_quality_floor: float = 0.60
    phase12_phase2_scheduler_cashout_soft_floor: float = 0.55
    phase12_phase2_scheduler_cashout_quality_floor: float = 0.78
    phase12_phase2_scheduler_cashout_cooldown_ticks: int = 2
    phase12_phase2_priority_lane_ttl: int = 2
    phase2_runtime_adaptive_enabled: bool = False
    phase2_runtime_low_pressure_min_hetero_ratio: float = 6.0
    phase2_runtime_high_pressure_min_hetero_ratio: float = 4.0
    phase2_runtime_low_pressure_min_pressure_ratio: float = 6.0
    phase2_runtime_high_pressure_min_pressure_ratio: float = 4.0
    phase2_runtime_low_pressure_min_long_prefill: int = 1024
    phase2_runtime_high_pressure_min_long_prefill: int = 768
    # Metrics
    enable_metrics_hook: bool = True
    metrics_short_request_tokens: int = 256
