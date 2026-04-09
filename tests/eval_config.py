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
    phase12_phase2_require_beneficiary_signal: bool = True,
    phase12_phase2_beneficiary_score_threshold: float = 0.55,
    phase2_enable_mixed_prefill_decode: bool = True,
    phase2_min_hetero_ratio: float = 2.0,
    phase2_min_long_prefill: int = 256,
    phase2_min_pressure_ratio: float = 2.0,
    phase2_execution_escape_mode: str = "bounded_spillover",
    phase2_execution_escape_spillover_cap: int = 3,
    phase2_execution_escape_max_active: int = 5,
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
    )
    phase2_kwargs = dict(
        phase2_enable_v1_true_unbind=False,
        phase2_enable_scheduler_cashout=False,
        phase2_enable_execution_escape=True,
        phase2_enable_mixed_prefill_decode=bool(phase2_enable_mixed_prefill_decode),
        phase2_min_hetero_ratio=float(phase2_min_hetero_ratio),
        phase2_min_long_prefill=int(phase2_min_long_prefill),
        phase2_min_pressure_ratio=float(phase2_min_pressure_ratio),
        phase2_execution_escape_mode=str(phase2_execution_escape_mode),
        phase2_execution_escape_spillover_cap=int(phase2_execution_escape_spillover_cap),
        phase2_execution_escape_max_active=int(phase2_execution_escape_max_active),
        phase2_dispatch_mode=phase2_dispatch_mode,
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
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=True,
            enable_sjf_reorder=True,
            queue_reorder_mode=str(queue_reorder_mode),
            queue_reorder_aging_quantum_us=float(queue_reorder_aging_quantum_us),
            phase2_consistency_mode="balanced",
            **phase1_kwargs,
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
        policy = WaveSlicePolicy(
            enable_phase1_scheduler=True,
            enable_phase2_modelrunner=True,
            enable_sjf_reorder=True,
            queue_reorder_mode=str(queue_reorder_mode),
            queue_reorder_aging_quantum_us=float(queue_reorder_aging_quantum_us),
            phase2_consistency_mode="strict",
            **phase1_kwargs,
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
        "phase1_ingress_target_chunk": args.phase1_ingress_target_chunk,
        "phase1_gamma": args.phase1_gamma,
        "phase1_ingress_direct_authoritative": args.phase1_ingress_direct_authoritative,
        "phase1_ingress_exact_chunk": args.phase1_ingress_exact_chunk,
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
        "phase2_execution_escape_mode": args.phase2_execution_escape_mode,
        "phase2_execution_escape_spillover_cap": args.phase2_execution_escape_spillover_cap,
        "phase2_execution_escape_max_active": args.phase2_execution_escape_max_active,
        "serialize_gpu_tests": args.serialize_gpu_tests,
        "gpu_lock_path": args.gpu_lock_path,
        "adapter_a": args.adapter_a,
        "adapter_b": args.adapter_b,
    }
