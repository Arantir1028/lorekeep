from __future__ import annotations

from argparse import Namespace
from typing import Any

from engine.vllm_hijacker import WaveSlicePolicy, inject_wave_slice, uninject_wave_slice


def configure_mode(
    *,
    model_name: str,
    mode: str,
    queue_reorder_mode: str,
    queue_reorder_aging_quantum_us: float,
    phase2_dispatch_mode: str,
    phase1_objective_mode: str,
    phase1_gamma: float,
    phase1_ingress_target_chunk: int,
    phase1_ingress_direct_authoritative: bool,
    phase1_ingress_exact_chunk: bool,
    phase12_phase2_gate_mode: str,
    phase12_phase2_soft_ratio_scale: float,
    phase12_phase2_soft_pressure_scale: float,
    phase12_phase2_soft_min_long_prefill: int,
    phase12_phase2_soft_allow_mixed_decode: bool,
    phase12_phase2_soft_recent_strength_floor: float,
    phase12_phase2_soft_require_cashout_signal: bool,
    phase12_phase2_soft_recent_chunk_match_scale: float,
    phase12_phase2_soft_window_score_threshold: float,
    phase12_phase2_soft_window_recent_weight: float,
    phase12_phase2_soft_window_chunk_weight: float,
    phase12_phase2_soft_window_pressure_weight: float,
    phase12_phase2_soft_window_ratio_weight: float,
    phase12_phase2_soft_window_decode_bonus: float,
    phase12_phase2_scheduler_cashout_soft_floor: float,
    phase12_phase2_scheduler_cashout_quality_floor: float,
    phase12_phase2_scheduler_cashout_cooldown_ticks: int,
    phase1_force_min_chunk: int = 128,
    phase1_target_long_fraction: float = 0.33,
    phase1_runtime_adaptive_enabled: bool = False,
    phase1_runtime_aggressive_long_fraction: float = 0.33,
    phase1_runtime_conservative_long_fraction: float = 0.50,
    phase1_runtime_aggressive_ingress_target_chunk: int = 768,
    phase1_runtime_conservative_ingress_target_chunk: int = 1536,
    phase1_runtime_queue_high_watermark: int = 8,
    phase1_runtime_waiting_short_high_watermark: int = 4,
    phase1_runtime_wait_us_high_watermark: float = 1_000_000.0,
    phase1_runtime_long_high_watermark: int = 3072,
    phase1_runtime_urgency_discount: float = 0.55,
    phase1_runtime_ema_alpha: float = 0.35,
    phase12_phase2_require_beneficiary_signal: bool = True,
    phase12_phase2_beneficiary_score_threshold: float = 0.55,
    phase2_enable_mixed_prefill_decode: bool = True,
    phase2_min_hetero_ratio: float = 2.0,
    phase2_min_long_prefill: int = 256,
    phase2_min_pressure_ratio: float = 2.0,
    phase2_enable_scheduler_cashout: bool = True,
    phase2_enable_execution_escape: bool = True,
    phase2_execution_escape_mode: str = "bounded_spillover",
    phase2_execution_escape_spillover_cap: int = 3,
    phase2_execution_escape_max_active: int = 5,
    phase2_runtime_adaptive_enabled: bool = False,
    phase2_runtime_low_pressure_min_hetero_ratio: float = 6.0,
    phase2_runtime_high_pressure_min_hetero_ratio: float = 4.0,
    phase2_runtime_low_pressure_min_pressure_ratio: float = 6.0,
    phase2_runtime_high_pressure_min_pressure_ratio: float = 4.0,
    phase2_runtime_low_pressure_min_long_prefill: int = 1024,
    phase2_runtime_high_pressure_min_long_prefill: int = 768,
    phase2_runtime_low_pressure_escape_spillover_cap: int = 1,
    phase2_runtime_high_pressure_escape_spillover_cap: int = 3,
    phase2_runtime_low_pressure_escape_max_active: int = 2,
    phase2_runtime_high_pressure_escape_max_active: int = 5,
    phase2_runtime_disable_execution_escape_below_pressure: float = -1.0,
) -> None:
    if mode == "baseline":
        uninject_wave_slice()
        return

    phase1_kwargs = dict(
        enable_metrics_hook=True,
        enable_tick_hide=False,
        enable_vllm_lora_compat_patch=True,
        scheduler_objective_mode=phase1_objective_mode,
        phase1_ingress_target_chunk=int(phase1_ingress_target_chunk),
        phase1_ingress_direct_authoritative=bool(phase1_ingress_direct_authoritative),
        phase1_ingress_exact_chunk=bool(phase1_ingress_exact_chunk),
        phase1_force_min_chunk=int(phase1_force_min_chunk),
        phase1_target_long_fraction=float(phase1_target_long_fraction),
        phase1_runtime_adaptive_enabled=bool(phase1_runtime_adaptive_enabled),
        phase1_runtime_aggressive_long_fraction=float(phase1_runtime_aggressive_long_fraction),
        phase1_runtime_conservative_long_fraction=float(phase1_runtime_conservative_long_fraction),
        phase1_runtime_aggressive_ingress_target_chunk=int(phase1_runtime_aggressive_ingress_target_chunk),
        phase1_runtime_conservative_ingress_target_chunk=int(phase1_runtime_conservative_ingress_target_chunk),
        phase1_runtime_queue_high_watermark=int(phase1_runtime_queue_high_watermark),
        phase1_runtime_waiting_short_high_watermark=int(phase1_runtime_waiting_short_high_watermark),
        phase1_runtime_wait_us_high_watermark=float(phase1_runtime_wait_us_high_watermark),
        phase1_runtime_long_high_watermark=int(phase1_runtime_long_high_watermark),
        phase1_runtime_urgency_discount=float(phase1_runtime_urgency_discount),
        phase1_runtime_ema_alpha=float(phase1_runtime_ema_alpha),
    )
    phase2_kwargs = dict(
        phase2_enable_scheduler_cashout=bool(phase2_enable_scheduler_cashout),
        phase2_enable_execution_escape=bool(phase2_enable_execution_escape),
        phase2_enable_mixed_prefill_decode=bool(phase2_enable_mixed_prefill_decode),
        phase2_min_hetero_ratio=float(phase2_min_hetero_ratio),
        phase2_min_long_prefill=int(phase2_min_long_prefill),
        phase2_min_pressure_ratio=float(phase2_min_pressure_ratio),
        phase2_execution_escape_mode=str(phase2_execution_escape_mode),
        phase2_execution_escape_spillover_cap=int(phase2_execution_escape_spillover_cap),
        phase2_execution_escape_max_active=int(phase2_execution_escape_max_active),
        phase2_dispatch_mode=phase2_dispatch_mode,
        phase2_runtime_adaptive_enabled=bool(phase2_runtime_adaptive_enabled),
        phase2_runtime_low_pressure_min_hetero_ratio=float(phase2_runtime_low_pressure_min_hetero_ratio),
        phase2_runtime_high_pressure_min_hetero_ratio=float(phase2_runtime_high_pressure_min_hetero_ratio),
        phase2_runtime_low_pressure_min_pressure_ratio=float(phase2_runtime_low_pressure_min_pressure_ratio),
        phase2_runtime_high_pressure_min_pressure_ratio=float(phase2_runtime_high_pressure_min_pressure_ratio),
        phase2_runtime_low_pressure_min_long_prefill=int(phase2_runtime_low_pressure_min_long_prefill),
        phase2_runtime_high_pressure_min_long_prefill=int(phase2_runtime_high_pressure_min_long_prefill),
        phase2_runtime_low_pressure_escape_spillover_cap=int(phase2_runtime_low_pressure_escape_spillover_cap),
        phase2_runtime_high_pressure_escape_spillover_cap=int(phase2_runtime_high_pressure_escape_spillover_cap),
        phase2_runtime_low_pressure_escape_max_active=int(phase2_runtime_low_pressure_escape_max_active),
        phase2_runtime_high_pressure_escape_max_active=int(phase2_runtime_high_pressure_escape_max_active),
        phase2_runtime_disable_execution_escape_below_pressure=float(phase2_runtime_disable_execution_escape_below_pressure),
    )
    phase12_kwargs = dict(
        phase12_phase2_gate_mode=str(phase12_phase2_gate_mode),
        phase12_phase2_soft_ratio_scale=float(phase12_phase2_soft_ratio_scale),
        phase12_phase2_soft_pressure_scale=float(phase12_phase2_soft_pressure_scale),
        phase12_phase2_soft_min_long_prefill=int(phase12_phase2_soft_min_long_prefill),
        phase12_phase2_soft_allow_mixed_decode=bool(phase12_phase2_soft_allow_mixed_decode),
        phase12_phase2_soft_recent_strength_floor=float(phase12_phase2_soft_recent_strength_floor),
        phase12_phase2_soft_require_cashout_signal=bool(phase12_phase2_soft_require_cashout_signal),
        phase12_phase2_soft_recent_chunk_match_scale=float(phase12_phase2_soft_recent_chunk_match_scale),
        phase12_phase2_soft_window_score_threshold=float(phase12_phase2_soft_window_score_threshold),
        phase12_phase2_soft_window_recent_weight=float(phase12_phase2_soft_window_recent_weight),
        phase12_phase2_soft_window_chunk_weight=float(phase12_phase2_soft_window_chunk_weight),
        phase12_phase2_soft_window_pressure_weight=float(phase12_phase2_soft_window_pressure_weight),
        phase12_phase2_soft_window_ratio_weight=float(phase12_phase2_soft_window_ratio_weight),
        phase12_phase2_soft_window_decode_bonus=float(phase12_phase2_soft_window_decode_bonus),
        phase12_phase2_scheduler_cashout_soft_floor=float(phase12_phase2_scheduler_cashout_soft_floor),
        phase12_phase2_scheduler_cashout_quality_floor=float(phase12_phase2_scheduler_cashout_quality_floor),
        phase12_phase2_scheduler_cashout_cooldown_ticks=int(phase12_phase2_scheduler_cashout_cooldown_ticks),
        phase12_phase2_require_beneficiary_signal=bool(phase12_phase2_require_beneficiary_signal),
        phase12_phase2_beneficiary_score_threshold=float(phase12_phase2_beneficiary_score_threshold),
    )

    if mode == "phase1_only":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=False,
            enable_sjf_reorder=True,
            queue_reorder_mode=str(queue_reorder_mode),
            queue_reorder_aging_quantum_us=float(queue_reorder_aging_quantum_us),
            **phase1_kwargs,
        )
        inject_wave_slice(model_name, gamma=float(phase1_gamma), policy=policy, force=True)
        return

    if mode == "phase1_lora_only":
        phase1_lora_kwargs = dict(phase1_kwargs)
        phase1_lora_kwargs["enable_tick_hide"] = True
        phase1_lora_kwargs["allow_phase1_tick_hide_with_lora"] = True
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=False,
            enable_sjf_reorder=True,
            queue_reorder_mode=str(queue_reorder_mode),
            queue_reorder_aging_quantum_us=float(queue_reorder_aging_quantum_us),
            **phase1_lora_kwargs,
        )
        inject_wave_slice(model_name, gamma=float(phase1_gamma), policy=policy, force=True)
        return

    if mode == "phase2_lora":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=False,
            enable_phase2_modelrunner=True,
            enable_sjf_reorder=False,
            phase2_consistency_mode="balanced",
            **phase2_kwargs,
            enable_metrics_hook=True,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
        )
        inject_wave_slice(model_name, policy=policy, force=True)
        return

    if mode == "phase12_lora":
        phase2_paths_enabled = bool(
            phase2_kwargs.get("phase2_enable_execution_escape")
            or phase2_kwargs.get("phase2_enable_scheduler_cashout")
        )
        phase12_phase1_kwargs = dict(phase1_kwargs)
        phase12_phase1_kwargs["enable_tick_hide"] = not phase2_paths_enabled
        phase12_phase1_kwargs["allow_phase1_tick_hide_with_lora"] = True
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=True,
            enable_sjf_reorder=True,
            queue_reorder_mode=str(queue_reorder_mode),
            queue_reorder_aging_quantum_us=float(queue_reorder_aging_quantum_us),
            phase2_consistency_mode="balanced",
            **phase12_phase1_kwargs,
            **phase2_kwargs,
            **phase12_kwargs,
        )
        inject_wave_slice(model_name, gamma=float(phase1_gamma), policy=policy, force=True)
        return

    if mode == "phase2_lora_strict":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=False,
            enable_phase2_modelrunner=True,
            enable_sjf_reorder=False,
            phase2_consistency_mode="strict",
            **phase2_kwargs,
            enable_metrics_hook=True,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
        )
        inject_wave_slice(model_name, policy=policy, force=True)
        return

    if mode == "phase12_lora_strict":
        phase2_paths_enabled = bool(
            phase2_kwargs.get("phase2_enable_execution_escape")
            or phase2_kwargs.get("phase2_enable_scheduler_cashout")
        )
        phase12_phase1_kwargs = dict(phase1_kwargs)
        phase12_phase1_kwargs["enable_tick_hide"] = not phase2_paths_enabled
        phase12_phase1_kwargs["allow_phase1_tick_hide_with_lora"] = True
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=True,
            enable_sjf_reorder=True,
            queue_reorder_mode=str(queue_reorder_mode),
            queue_reorder_aging_quantum_us=float(queue_reorder_aging_quantum_us),
            phase2_consistency_mode="strict",
            **phase12_phase1_kwargs,
            **phase2_kwargs,
            **phase12_kwargs,
        )
        inject_wave_slice(model_name, gamma=float(phase1_gamma), policy=policy, force=True)
        return

    if mode == "baseline_lora_compat":
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=False,
            enable_phase2_modelrunner=False,
            enable_metrics_hook=True,
            enable_sjf_reorder=False,
            enable_tick_hide=False,
            enable_vllm_lora_compat_patch=True,
        )
        inject_wave_slice(model_name, policy=policy, force=True)
        return

    raise ValueError(f"Unknown mode: {mode}")


def build_summary_config(args: Namespace, *, short_a_repeat: int, short_b_repeat: int) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "max_new_tokens": args.max_new_tokens,
        "timeout_sec": args.timeout_sec,
        "warmup_iters": args.warmup_iters,
        "repeats": args.repeats,
        "short_repeat": args.short_repeat,
        "short_a_repeat": short_a_repeat,
        "short_b_repeat": short_b_repeat,
        "long_repeat": args.long_repeat,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_partial_prefills": getattr(args, "max_num_partial_prefills", 1),
        "max_long_partial_prefills": getattr(args, "max_long_partial_prefills", 1),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "requests_json": args.requests_json,
        "lora_requests_json": args.lora_requests_json,
        "queue_reorder_mode": args.queue_reorder_mode,
        "queue_reorder_aging_quantum_us": args.queue_reorder_aging_quantum_us,
        "phase1_objective_mode": args.phase1_objective_mode,
        "phase1_baseline_mode": args.phase1_baseline_mode,
        "include_phase12": args.include_phase12,
        "include_strict": args.include_strict,
        "include_phase1_lora_only": getattr(args, "include_phase1_lora_only", False),
        "phase1_ingress_target_chunk": args.phase1_ingress_target_chunk,
        "phase1_gamma": args.phase1_gamma,
        "phase1_ingress_direct_authoritative": args.phase1_ingress_direct_authoritative,
        "phase1_ingress_exact_chunk": args.phase1_ingress_exact_chunk,
        "phase1_force_min_chunk": getattr(args, "phase1_force_min_chunk", 128),
        "phase1_target_long_fraction": getattr(args, "phase1_target_long_fraction", 0.33),
        "phase1_runtime_adaptive_enabled": getattr(args, "phase1_runtime_adaptive_enabled", False),
        "phase1_runtime_aggressive_long_fraction": getattr(args, "phase1_runtime_aggressive_long_fraction", 0.33),
        "phase1_runtime_conservative_long_fraction": getattr(args, "phase1_runtime_conservative_long_fraction", 0.50),
        "phase1_runtime_aggressive_ingress_target_chunk": getattr(args, "phase1_runtime_aggressive_ingress_target_chunk", 768),
        "phase1_runtime_conservative_ingress_target_chunk": getattr(args, "phase1_runtime_conservative_ingress_target_chunk", 1536),
        "phase1_runtime_queue_high_watermark": getattr(args, "phase1_runtime_queue_high_watermark", 8),
        "phase1_runtime_waiting_short_high_watermark": getattr(args, "phase1_runtime_waiting_short_high_watermark", 4),
        "phase1_runtime_wait_us_high_watermark": getattr(args, "phase1_runtime_wait_us_high_watermark", 1_000_000.0),
        "phase1_runtime_long_high_watermark": getattr(args, "phase1_runtime_long_high_watermark", 3072),
        "phase1_runtime_urgency_discount": getattr(args, "phase1_runtime_urgency_discount", 0.55),
        "phase1_runtime_ema_alpha": getattr(args, "phase1_runtime_ema_alpha", 0.35),
        "phase12_phase2_gate_mode": args.phase12_phase2_gate_mode,
        "phase12_phase2_soft_ratio_scale": args.phase12_phase2_soft_ratio_scale,
        "phase12_phase2_soft_pressure_scale": args.phase12_phase2_soft_pressure_scale,
        "phase12_phase2_soft_min_long_prefill": args.phase12_phase2_soft_min_long_prefill,
        "phase12_phase2_soft_allow_mixed_decode": args.phase12_phase2_soft_allow_mixed_decode,
        "phase12_phase2_soft_recent_strength_floor": args.phase12_phase2_soft_recent_strength_floor,
        "phase12_phase2_soft_require_cashout_signal": args.phase12_phase2_soft_require_cashout_signal,
        "phase12_phase2_soft_recent_chunk_match_scale": args.phase12_phase2_soft_recent_chunk_match_scale,
        "phase12_phase2_soft_window_score_threshold": args.phase12_phase2_soft_window_score_threshold,
        "phase12_phase2_soft_window_recent_weight": args.phase12_phase2_soft_window_recent_weight,
        "phase12_phase2_soft_window_chunk_weight": args.phase12_phase2_soft_window_chunk_weight,
        "phase12_phase2_soft_window_pressure_weight": args.phase12_phase2_soft_window_pressure_weight,
        "phase12_phase2_soft_window_ratio_weight": args.phase12_phase2_soft_window_ratio_weight,
        "phase12_phase2_soft_window_decode_bonus": args.phase12_phase2_soft_window_decode_bonus,
        "phase12_phase2_scheduler_cashout_soft_floor": getattr(
            args, "phase12_phase2_scheduler_cashout_soft_floor", 0.55
        ),
        "phase12_phase2_scheduler_cashout_quality_floor": getattr(
            args, "phase12_phase2_scheduler_cashout_quality_floor", 0.78
        ),
        "phase12_phase2_scheduler_cashout_cooldown_ticks": getattr(
            args, "phase12_phase2_scheduler_cashout_cooldown_ticks", 2
        ),
        "phase12_phase2_require_beneficiary_signal": getattr(
            args, "phase12_phase2_require_beneficiary_signal", True
        ),
        "phase12_phase2_beneficiary_score_threshold": getattr(
            args, "phase12_phase2_beneficiary_score_threshold", 0.55
        ),
        "phase2_dispatch_mode": args.phase2_dispatch_mode,
        "phase2_enable_mixed_prefill_decode": args.phase2_enable_mixed_prefill_decode,
        "phase2_min_hetero_ratio": args.phase2_min_hetero_ratio,
        "phase2_min_long_prefill": args.phase2_min_long_prefill,
        "phase2_min_pressure_ratio": args.phase2_min_pressure_ratio,
        "phase2_baseline_enable_chunked_prefill": getattr(args, "phase2_baseline_enable_chunked_prefill", True),
        "phase2_enable_scheduler_cashout": args.phase2_enable_scheduler_cashout,
        "phase2_enable_execution_escape": args.phase2_enable_execution_escape,
        "phase2_execution_escape_mode": args.phase2_execution_escape_mode,
        "phase2_execution_escape_spillover_cap": args.phase2_execution_escape_spillover_cap,
        "phase2_execution_escape_max_active": args.phase2_execution_escape_max_active,
        "phase2_runtime_adaptive_enabled": getattr(args, "phase2_runtime_adaptive_enabled", False),
        "phase2_runtime_low_pressure_min_hetero_ratio": getattr(args, "phase2_runtime_low_pressure_min_hetero_ratio", 6.0),
        "phase2_runtime_high_pressure_min_hetero_ratio": getattr(args, "phase2_runtime_high_pressure_min_hetero_ratio", 4.0),
        "phase2_runtime_low_pressure_min_pressure_ratio": getattr(args, "phase2_runtime_low_pressure_min_pressure_ratio", 6.0),
        "phase2_runtime_high_pressure_min_pressure_ratio": getattr(args, "phase2_runtime_high_pressure_min_pressure_ratio", 4.0),
        "phase2_runtime_low_pressure_min_long_prefill": getattr(args, "phase2_runtime_low_pressure_min_long_prefill", 1024),
        "phase2_runtime_high_pressure_min_long_prefill": getattr(args, "phase2_runtime_high_pressure_min_long_prefill", 768),
        "phase2_runtime_low_pressure_escape_spillover_cap": getattr(args, "phase2_runtime_low_pressure_escape_spillover_cap", 1),
        "phase2_runtime_high_pressure_escape_spillover_cap": getattr(args, "phase2_runtime_high_pressure_escape_spillover_cap", 3),
        "phase2_runtime_low_pressure_escape_max_active": getattr(args, "phase2_runtime_low_pressure_escape_max_active", 2),
        "phase2_runtime_high_pressure_escape_max_active": getattr(args, "phase2_runtime_high_pressure_escape_max_active", 5),
        "phase2_runtime_disable_execution_escape_below_pressure": getattr(args, "phase2_runtime_disable_execution_escape_below_pressure", -1.0),
        "serialize_gpu_tests": args.serialize_gpu_tests,
        "gpu_lock_path": args.gpu_lock_path,
        "adapter_a": args.adapter_a,
        "adapter_b": args.adapter_b,
    }
