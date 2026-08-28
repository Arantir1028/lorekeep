from __future__ import annotations

from argparse import Namespace
from typing import Any

from waveslice import WaveSliceConfig, WaveSlicePolicy

_MODES = {
    "phase1_only",
    "phase2_lora",
    "phase12_lora",
    "baseline_lora_compat",
}


def build_wave_slice_config(
    *, model_name: str, mode: str, **options: Any
) -> WaveSliceConfig | None:
    if mode == "baseline":
        return None
    if mode not in _MODES:
        raise ValueError(f"unknown mode: {mode}")
    phase1 = mode.startswith("phase1") or mode.startswith("phase12")
    phase2 = mode.startswith("phase2") or mode.startswith("phase12")
    values = {
        name: value
        for name, value in options.items()
        if name in WaveSlicePolicy.__dataclass_fields__
    }
    values.update(
        {
            "scheduler_objective_mode": options.get("phase1_objective_mode", "fair_escape"),
            "enable_metrics_hook": True,
            "enable_phase1_scheduler": phase1,
            "enable_phase2_scheduler": phase2,
            "enable_sjf_reorder": phase1,
        }
    )
    return WaveSliceConfig(
        lut_model=model_name,
        gamma=float(options.get("phase1_gamma", 2.0)),
        policy=WaveSlicePolicy(**values),
    )


_SUMMARY_FIELDS = (
    "model_name",
    "model_path",
    "max_new_tokens",
    "timeout_sec",
    "warmup_iters",
    "repeats",
    "short_repeat",
    "long_repeat",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_partial_prefills",
    "max_long_partial_prefills",
    "gpu_memory_utilization",
    "trust_remote_code",
    "requests_json",
    "lora_requests_json",
    "queue_reorder_mode",
    "queue_reorder_aging_quantum_us",
    "phase1_objective_mode",
    "phase1_baseline_mode",
    "include_phase12",
    "phase1_ingress_target_chunk",
    "phase1_gamma",
    "phase1_ingress_direct_authoritative",
    "phase1_ingress_exact_chunk",
    "phase1_force_min_chunk",
    "phase1_target_long_fraction",
    "phase1_runtime_adaptive_enabled",
    "phase1_runtime_aggressive_long_fraction",
    "phase1_runtime_conservative_long_fraction",
    "phase1_runtime_aggressive_ingress_target_chunk",
    "phase1_runtime_conservative_ingress_target_chunk",
    "phase1_runtime_queue_high_watermark",
    "phase1_runtime_waiting_short_high_watermark",
    "phase1_runtime_wait_us_high_watermark",
    "phase1_runtime_long_high_watermark",
    "phase1_runtime_urgency_discount",
    "phase1_runtime_ema_alpha",
    "phase12_phase2_gate_mode",
    "phase12_phase2_soft_min_long_prefill",
    "phase12_phase2_soft_allow_mixed_decode",
    "phase12_phase2_scheduler_cashout_soft_floor",
    "phase12_phase2_scheduler_cashout_quality_floor",
    "phase12_phase2_scheduler_cashout_cooldown_ticks",
    "phase12_phase2_beneficiary_score_threshold",
    "phase2_dispatch_mode",
    "phase2_min_hetero_ratio",
    "phase2_min_long_prefill",
    "phase2_min_pressure_ratio",
    "phase2_baseline_enable_chunked_prefill",
    "phase2_enable_scheduler_cashout",
    "phase2_runtime_adaptive_enabled",
    "phase2_runtime_low_pressure_min_hetero_ratio",
    "phase2_runtime_high_pressure_min_hetero_ratio",
    "phase2_runtime_low_pressure_min_pressure_ratio",
    "phase2_runtime_high_pressure_min_pressure_ratio",
    "phase2_runtime_low_pressure_min_long_prefill",
    "phase2_runtime_high_pressure_min_long_prefill",
    "serialize_gpu_tests",
    "gpu_lock_path",
    "adapter_a",
    "adapter_b",
)
_SUMMARY_DEFAULTS = {
    "max_num_partial_prefills": 1,
    "max_long_partial_prefills": 1,
    "phase12_phase2_scheduler_cashout_soft_floor": 0.55,
    "phase12_phase2_scheduler_cashout_quality_floor": 0.78,
    "phase12_phase2_scheduler_cashout_cooldown_ticks": 2,
    "phase12_phase2_beneficiary_score_threshold": 0.55,
}


def build_summary_config(
    args: Namespace, *, short_a_repeat: int, short_b_repeat: int
) -> dict[str, Any]:
    return {
        **{field: getattr(args, field, _SUMMARY_DEFAULTS.get(field)) for field in _SUMMARY_FIELDS},
        "short_a_repeat": short_a_repeat,
        "short_b_repeat": short_b_repeat,
    }
