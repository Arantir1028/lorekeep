"""Repeated Wave-Slice claim evaluation with warmup and error-rate reporting.

What this script measures:
1) Phase-I slicing gain (non-LoRA): TTFT p99 gain + text mismatch rate.
2) Phase-II multi-stream gain (LoRA): TTFT/slowdown gain + text mismatch rate.
3) Baseline LoRA noise floor: baseline-vs-baseline mismatch rate.

The script is designed for paper-style experiments:
- warmup iterations
- many repeats
- per-repeat records + aggregate statistics
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

from engine.runtime_bootstrap import bootstrap_vllm_runtime
from engine.vllm_hijacker import (
    get_wave_slice_metrics,
    reset_wave_slice_metrics,
    uninject_wave_slice,
)
from experiments.model_assets import ensure_adapters
from config.experiment_catalog import safe_key
from tools.experiment_lock import gpu_experiment_lock
from eval_config import build_summary_config, configure_mode as _configure_mode
from eval_output import build_summary, print_summary, write_summary_json
from eval_support import (
    Req,
    bool_arg_from_argv,
    fit_lora_requests_to_context,
    fit_requests_to_context,
    load_reqs_json,
    measure_input_tokens,
    percentile as _percentile,
    ratio as _ratio,
    run_phase1_pair,
    run_phase2_block,
    stats as _stats,
    str_arg_from_argv,
)

bootstrap_vllm_runtime()


def _ensure_eval_adapters(
    *,
    model_path: str,
    adapters_root: str,
    trust_remote_code: bool,
) -> tuple[str, str]:
    model_key = safe_key(model_path)
    out_dir = os.path.join(adapters_root, model_key)
    os.makedirs(out_dir, exist_ok=True)
    return ensure_adapters(
        base_model_path=model_path,
        out_dir=out_dir,
        trust_remote_code=trust_remote_code,
    )


def _mk_lora_request(LoRARequest: Any, name: str, req_id: int, path: str) -> Any:
    try:
        return LoRARequest(lora_name=name, lora_int_id=req_id, lora_path=path)
    except TypeError:
        return LoRARequest(lora_name=name, lora_int_id=req_id, lora_local_path=path)
def _cleanup_engine(engine: Optional[Any]) -> None:
    if engine is not None:
        del engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    uninject_wave_slice()


def _build_engine(
    *,
    model_path: str,
    model_name: str,
    mode: str,
    enable_lora: bool,
    max_num_batched_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    queue_reorder_mode: str,
    queue_reorder_aging_quantum_us: float,
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
    phase12_phase2_require_beneficiary_signal: bool,
    phase12_phase2_beneficiary_score_threshold: float,
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
    max_num_partial_prefills: int = 1,
    max_long_partial_prefills: int = 1,
    enable_chunked_prefill: bool,
    adapter_a: Optional[str] = None,
    adapter_b: Optional[str] = None,
    phase2_dispatch_mode: str = "synchronized",
    trust_remote_code: bool = False,
) -> tuple[Any, dict[str, Any]]:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine

    _configure_mode(
        model_name=model_name,
        mode=mode,
        queue_reorder_mode=queue_reorder_mode,
        queue_reorder_aging_quantum_us=queue_reorder_aging_quantum_us,
        phase2_dispatch_mode=phase2_dispatch_mode,
        phase1_objective_mode=phase1_objective_mode,
        phase1_gamma=phase1_gamma,
        phase1_ingress_target_chunk=phase1_ingress_target_chunk,
        phase1_ingress_direct_authoritative=phase1_ingress_direct_authoritative,
        phase1_ingress_exact_chunk=phase1_ingress_exact_chunk,
        phase1_force_min_chunk=phase1_force_min_chunk,
        phase1_target_long_fraction=phase1_target_long_fraction,
        phase1_runtime_adaptive_enabled=phase1_runtime_adaptive_enabled,
        phase1_runtime_aggressive_long_fraction=phase1_runtime_aggressive_long_fraction,
        phase1_runtime_conservative_long_fraction=phase1_runtime_conservative_long_fraction,
        phase1_runtime_aggressive_ingress_target_chunk=phase1_runtime_aggressive_ingress_target_chunk,
        phase1_runtime_conservative_ingress_target_chunk=phase1_runtime_conservative_ingress_target_chunk,
        phase1_runtime_queue_high_watermark=phase1_runtime_queue_high_watermark,
        phase1_runtime_waiting_short_high_watermark=phase1_runtime_waiting_short_high_watermark,
        phase1_runtime_wait_us_high_watermark=phase1_runtime_wait_us_high_watermark,
        phase1_runtime_long_high_watermark=phase1_runtime_long_high_watermark,
        phase1_runtime_urgency_discount=phase1_runtime_urgency_discount,
        phase1_runtime_ema_alpha=phase1_runtime_ema_alpha,
        phase12_phase2_gate_mode=phase12_phase2_gate_mode,
        phase12_phase2_soft_ratio_scale=phase12_phase2_soft_ratio_scale,
        phase12_phase2_soft_pressure_scale=phase12_phase2_soft_pressure_scale,
        phase12_phase2_soft_min_long_prefill=phase12_phase2_soft_min_long_prefill,
        phase12_phase2_soft_allow_mixed_decode=phase12_phase2_soft_allow_mixed_decode,
        phase12_phase2_soft_recent_strength_floor=phase12_phase2_soft_recent_strength_floor,
        phase12_phase2_soft_require_cashout_signal=phase12_phase2_soft_require_cashout_signal,
        phase12_phase2_soft_recent_chunk_match_scale=phase12_phase2_soft_recent_chunk_match_scale,
        phase12_phase2_soft_window_score_threshold=phase12_phase2_soft_window_score_threshold,
        phase12_phase2_soft_window_recent_weight=phase12_phase2_soft_window_recent_weight,
        phase12_phase2_soft_window_chunk_weight=phase12_phase2_soft_window_chunk_weight,
        phase12_phase2_soft_window_pressure_weight=phase12_phase2_soft_window_pressure_weight,
        phase12_phase2_soft_window_ratio_weight=phase12_phase2_soft_window_ratio_weight,
        phase12_phase2_soft_window_decode_bonus=phase12_phase2_soft_window_decode_bonus,
        phase12_phase2_scheduler_cashout_soft_floor=phase12_phase2_scheduler_cashout_soft_floor,
        phase12_phase2_scheduler_cashout_quality_floor=phase12_phase2_scheduler_cashout_quality_floor,
        phase12_phase2_scheduler_cashout_cooldown_ticks=phase12_phase2_scheduler_cashout_cooldown_ticks,
        phase12_phase2_require_beneficiary_signal=phase12_phase2_require_beneficiary_signal,
        phase12_phase2_beneficiary_score_threshold=phase12_phase2_beneficiary_score_threshold,
        phase2_enable_mixed_prefill_decode=phase2_enable_mixed_prefill_decode,
        phase2_min_hetero_ratio=phase2_min_hetero_ratio,
        phase2_min_long_prefill=phase2_min_long_prefill,
        phase2_min_pressure_ratio=phase2_min_pressure_ratio,
        phase2_enable_scheduler_cashout=phase2_enable_scheduler_cashout,
        phase2_enable_execution_escape=phase2_enable_execution_escape,
        phase2_execution_escape_mode=phase2_execution_escape_mode,
        phase2_execution_escape_spillover_cap=phase2_execution_escape_spillover_cap,
        phase2_execution_escape_max_active=phase2_execution_escape_max_active,
        phase2_runtime_adaptive_enabled=phase2_runtime_adaptive_enabled,
        phase2_runtime_low_pressure_min_hetero_ratio=phase2_runtime_low_pressure_min_hetero_ratio,
        phase2_runtime_high_pressure_min_hetero_ratio=phase2_runtime_high_pressure_min_hetero_ratio,
        phase2_runtime_low_pressure_min_pressure_ratio=phase2_runtime_low_pressure_min_pressure_ratio,
        phase2_runtime_high_pressure_min_pressure_ratio=phase2_runtime_high_pressure_min_pressure_ratio,
        phase2_runtime_low_pressure_min_long_prefill=phase2_runtime_low_pressure_min_long_prefill,
        phase2_runtime_high_pressure_min_long_prefill=phase2_runtime_high_pressure_min_long_prefill,
        phase2_runtime_low_pressure_escape_spillover_cap=phase2_runtime_low_pressure_escape_spillover_cap,
        phase2_runtime_high_pressure_escape_spillover_cap=phase2_runtime_high_pressure_escape_spillover_cap,
        phase2_runtime_low_pressure_escape_max_active=phase2_runtime_low_pressure_escape_max_active,
        phase2_runtime_high_pressure_escape_max_active=phase2_runtime_high_pressure_escape_max_active,
        phase2_runtime_disable_execution_escape_below_pressure=phase2_runtime_disable_execution_escape_below_pressure,
    )

    effective_batched_tokens = int(max_num_batched_tokens)
    if not enable_chunked_prefill:
        effective_batched_tokens = max(effective_batched_tokens, int(max_model_len))

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=trust_remote_code,
        seed=0,
        enable_lora=enable_lora,
        max_loras=(
            max(2, len([p for p in (adapter_a, adapter_b) if p]))
            if enable_lora
            else 1
        ),
        max_lora_rank=32,
        max_num_batched_tokens=effective_batched_tokens,
        max_num_partial_prefills=max(1, int(max_num_partial_prefills)),
        max_long_partial_prefills=max(1, int(max_long_partial_prefills)),
        enable_chunked_prefill=enable_chunked_prefill,
        disable_sliding_window=True,
        enforce_eager=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    lora_map: dict[str, Any] = {}
    if enable_lora:
        from vllm.lora.request import LoRARequest

        if not adapter_a or not adapter_b:
            raise ValueError("LoRA mode requires adapter_a and adapter_b.")
        lora_map = {
            "A": _mk_lora_request(LoRARequest, "adapter_A", 1, adapter_a),
            "B": _mk_lora_request(LoRARequest, "adapter_B", 2, adapter_b),
        }
    return engine, lora_map


def _run_round(
    *,
    engine: Any,
    reqs: list[Req],
    max_new_tokens: int,
    ignore_eos: bool,
    timeout_sec: int,
    enable_lora: bool,
    lora_map: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    from vllm.sampling_params import SamplingParams

    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=ignore_eos)
    trackers: dict[str, dict[str, Any]] = {}
    reset_wave_slice_metrics()

    pending_reqs = sorted(
        reqs,
        key=lambda r: (float(getattr(r, "arrival_offset_s", 0.0) or 0.0), str(r.req_id)),
    )
    next_idx = 0
    output_event_index = 0
    round_start = time.perf_counter()
    deadline = time.time() + timeout_sec
    while time.time() < deadline and (
        next_idx < len(pending_reqs) or engine.has_unfinished_requests()
    ):
        now = time.perf_counter()
        elapsed_s = now - round_start
        while next_idx < len(pending_reqs):
            r = pending_reqs[next_idx]
            if float(getattr(r, "arrival_offset_s", 0.0) or 0.0) > elapsed_s:
                break
            rid = f"{run_tag}:{r.req_id}"
            if enable_lora:
                engine.add_request(rid, r.prompt, sampling, lora_request=lora_map[r.lora_tag or "A"])
            else:
                engine.add_request(rid, r.prompt, sampling)
            trackers[rid] = {
                "orig_req_id": r.req_id,
                "arrival_s": now,
                "round_start_s": round_start,
                "scheduled_arrival_offset_s": float(getattr(r, "arrival_offset_s", 0.0) or 0.0),
                "first_s": None,
                "finish_s": None,
                "is_short": r.is_short,
                "text": "",
            }
            next_idx += 1

        if not engine.has_unfinished_requests():
            if next_idx < len(pending_reqs):
                next_arrival = float(getattr(pending_reqs[next_idx], "arrival_offset_s", 0.0) or 0.0)
                sleep_s = max(0.0, min(0.01, next_arrival - (time.perf_counter() - round_start)))
                if sleep_s > 0:
                    time.sleep(sleep_s)
            continue

        outputs = engine.step()
        now = time.perf_counter()
        for out in outputs:
            output_event_index += 1
            rid = out.request_id
            if rid not in trackers:
                continue
            tok_count = 0
            txt = ""
            try:
                payload = out.outputs[0]
                tok_count = len(payload.token_ids)
                txt = str(payload.text or "")
            except Exception:
                pass
            if tok_count > 0 and trackers[rid]["first_s"] is None:
                trackers[rid]["first_s"] = now
                trackers[rid]["first_event_index"] = int(output_event_index)
            if out.finished:
                trackers[rid]["finish_s"] = now
                trackers[rid]["text"] = txt
                trackers[rid]["finish_event_index"] = int(output_event_index)
    round_end = time.perf_counter()

    ttft_short_ms: list[float] = []
    finished_count = 0
    for tr in trackers.values():
        if tr["is_short"] and tr["first_s"] is not None:
            ttft_short_ms.append((tr["first_s"] - tr["arrival_s"]) * 1000.0)
        if tr["finish_s"] is not None:
            finished_count += 1

    report = get_wave_slice_metrics(reset=True)
    result = {
        "texts": {tr["orig_req_id"]: tr["text"] for tr in trackers.values()},
        "request_timings": {
            tr["orig_req_id"]: {
                "arrival_offset_s": float(tr["scheduled_arrival_offset_s"]),
                "first_latency_ms": ((tr["first_s"] - tr["arrival_s"]) * 1000.0) if tr["first_s"] is not None else None,
                "finish_latency_ms": ((tr["finish_s"] - tr["arrival_s"]) * 1000.0) if tr["finish_s"] is not None else None,
                "scheduled_first_latency_ms": (
                    (tr["first_s"] - (tr["round_start_s"] + tr["scheduled_arrival_offset_s"])) * 1000.0
                ) if tr["first_s"] is not None else None,
                "scheduled_finish_latency_ms": (
                    (tr["finish_s"] - (tr["round_start_s"] + tr["scheduled_arrival_offset_s"])) * 1000.0
                ) if tr["finish_s"] is not None else None,
                "first_event_index": tr.get("first_event_index"),
                "finish_event_index": tr.get("finish_event_index"),
                "is_short": bool(tr["is_short"]),
            }
            for tr in trackers.values()
        },
        "ttft_short_p99_ms": _percentile(ttft_short_ms, 99.0),
        "round_wall_ms": (round_end - round_start) * 1000.0,
        "timed_out": finished_count != len(trackers),
        "finished_requests": finished_count,
        "total_requests": len(trackers),
        "max_arrival_offset_s": max((float(getattr(r, "arrival_offset_s", 0.0) or 0.0) for r in pending_reqs), default=0.0),
        "hook_report": report,
    }
    return result


def _run_mode_series(
    *,
    model_path: str,
    model_name: str,
    reqs: list[Req],
    max_new_tokens: int,
    ignore_eos: bool,
    timeout_sec: int,
    mode: str,
    enable_lora: bool,
    warmup_iters: int,
    repeats: int,
    max_num_batched_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    queue_reorder_mode: str,
    queue_reorder_aging_quantum_us: float,
    phase1_objective_mode: str,
    phase1_gamma: float,
    phase1_ingress_target_chunk: int,
    phase1_ingress_direct_authoritative: bool,
    phase1_ingress_exact_chunk: bool,
    phase1_force_min_chunk: int = 128,
    phase1_target_long_fraction: float = 0.33,
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
    phase12_phase2_scheduler_cashout_soft_floor: float = 0.55,
    phase12_phase2_scheduler_cashout_quality_floor: float = 0.78,
    phase12_phase2_scheduler_cashout_cooldown_ticks: int = 2,
    phase12_phase2_require_beneficiary_signal: bool = True,
    phase12_phase2_beneficiary_score_threshold: float = 0.55,
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
    max_num_partial_prefills: int = 1,
    max_long_partial_prefills: int = 1,
    enable_chunked_prefill: bool,
    adapter_a: Optional[str] = None,
    adapter_b: Optional[str] = None,
    phase2_dispatch_mode: str = "synchronized",
    trust_remote_code: bool = False,
) -> list[dict[str, Any]]:
    engine = None
    try:
        engine, lora_map = _build_engine(
            model_path=model_path,
            model_name=model_name,
            mode=mode,
            enable_lora=enable_lora,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            queue_reorder_mode=queue_reorder_mode,
            queue_reorder_aging_quantum_us=queue_reorder_aging_quantum_us,
            phase1_objective_mode=phase1_objective_mode,
            phase1_gamma=phase1_gamma,
            phase1_ingress_target_chunk=phase1_ingress_target_chunk,
            phase1_ingress_direct_authoritative=phase1_ingress_direct_authoritative,
            phase1_ingress_exact_chunk=phase1_ingress_exact_chunk,
            phase1_force_min_chunk=phase1_force_min_chunk,
            phase1_target_long_fraction=phase1_target_long_fraction,
            phase1_runtime_adaptive_enabled=phase1_runtime_adaptive_enabled,
            phase1_runtime_aggressive_long_fraction=phase1_runtime_aggressive_long_fraction,
            phase1_runtime_conservative_long_fraction=phase1_runtime_conservative_long_fraction,
            phase1_runtime_aggressive_ingress_target_chunk=phase1_runtime_aggressive_ingress_target_chunk,
            phase1_runtime_conservative_ingress_target_chunk=phase1_runtime_conservative_ingress_target_chunk,
            phase1_runtime_queue_high_watermark=phase1_runtime_queue_high_watermark,
            phase1_runtime_waiting_short_high_watermark=phase1_runtime_waiting_short_high_watermark,
            phase1_runtime_wait_us_high_watermark=phase1_runtime_wait_us_high_watermark,
            phase1_runtime_long_high_watermark=phase1_runtime_long_high_watermark,
            phase1_runtime_urgency_discount=phase1_runtime_urgency_discount,
            phase1_runtime_ema_alpha=phase1_runtime_ema_alpha,
            phase12_phase2_gate_mode=phase12_phase2_gate_mode,
            phase12_phase2_soft_ratio_scale=phase12_phase2_soft_ratio_scale,
            phase12_phase2_soft_pressure_scale=phase12_phase2_soft_pressure_scale,
            phase12_phase2_soft_min_long_prefill=phase12_phase2_soft_min_long_prefill,
            phase12_phase2_soft_allow_mixed_decode=phase12_phase2_soft_allow_mixed_decode,
            phase12_phase2_soft_recent_strength_floor=phase12_phase2_soft_recent_strength_floor,
            phase12_phase2_soft_require_cashout_signal=phase12_phase2_soft_require_cashout_signal,
            phase12_phase2_soft_recent_chunk_match_scale=phase12_phase2_soft_recent_chunk_match_scale,
            phase12_phase2_soft_window_score_threshold=phase12_phase2_soft_window_score_threshold,
            phase12_phase2_soft_window_recent_weight=phase12_phase2_soft_window_recent_weight,
            phase12_phase2_soft_window_chunk_weight=phase12_phase2_soft_window_chunk_weight,
            phase12_phase2_soft_window_pressure_weight=phase12_phase2_soft_window_pressure_weight,
            phase12_phase2_soft_window_ratio_weight=phase12_phase2_soft_window_ratio_weight,
            phase12_phase2_soft_window_decode_bonus=phase12_phase2_soft_window_decode_bonus,
            phase12_phase2_scheduler_cashout_soft_floor=phase12_phase2_scheduler_cashout_soft_floor,
            phase12_phase2_scheduler_cashout_quality_floor=phase12_phase2_scheduler_cashout_quality_floor,
            phase12_phase2_scheduler_cashout_cooldown_ticks=phase12_phase2_scheduler_cashout_cooldown_ticks,
            phase12_phase2_require_beneficiary_signal=phase12_phase2_require_beneficiary_signal,
            phase12_phase2_beneficiary_score_threshold=phase12_phase2_beneficiary_score_threshold,
            phase2_enable_mixed_prefill_decode=phase2_enable_mixed_prefill_decode,
            phase2_min_hetero_ratio=phase2_min_hetero_ratio,
            phase2_min_long_prefill=phase2_min_long_prefill,
            phase2_min_pressure_ratio=phase2_min_pressure_ratio,
            phase2_enable_scheduler_cashout=phase2_enable_scheduler_cashout,
            phase2_enable_execution_escape=phase2_enable_execution_escape,
            phase2_execution_escape_mode=phase2_execution_escape_mode,
            phase2_execution_escape_spillover_cap=phase2_execution_escape_spillover_cap,
            phase2_execution_escape_max_active=phase2_execution_escape_max_active,
            phase2_runtime_adaptive_enabled=phase2_runtime_adaptive_enabled,
            phase2_runtime_low_pressure_min_hetero_ratio=phase2_runtime_low_pressure_min_hetero_ratio,
            phase2_runtime_high_pressure_min_hetero_ratio=phase2_runtime_high_pressure_min_hetero_ratio,
            phase2_runtime_low_pressure_min_pressure_ratio=phase2_runtime_low_pressure_min_pressure_ratio,
            phase2_runtime_high_pressure_min_pressure_ratio=phase2_runtime_high_pressure_min_pressure_ratio,
            phase2_runtime_low_pressure_min_long_prefill=phase2_runtime_low_pressure_min_long_prefill,
            phase2_runtime_high_pressure_min_long_prefill=phase2_runtime_high_pressure_min_long_prefill,
            phase2_runtime_low_pressure_escape_spillover_cap=phase2_runtime_low_pressure_escape_spillover_cap,
            phase2_runtime_high_pressure_escape_spillover_cap=phase2_runtime_high_pressure_escape_spillover_cap,
            phase2_runtime_low_pressure_escape_max_active=phase2_runtime_low_pressure_escape_max_active,
            phase2_runtime_high_pressure_escape_max_active=phase2_runtime_high_pressure_escape_max_active,
            phase2_runtime_disable_execution_escape_below_pressure=phase2_runtime_disable_execution_escape_below_pressure,
            max_num_partial_prefills=max_num_partial_prefills,
            max_long_partial_prefills=max_long_partial_prefills,
            enable_chunked_prefill=enable_chunked_prefill,
            adapter_a=adapter_a,
            adapter_b=adapter_b,
            phase2_dispatch_mode=phase2_dispatch_mode,
            trust_remote_code=trust_remote_code,
        )
        for i in range(warmup_iters):
            _run_round(
                engine=engine,
                reqs=reqs,
                max_new_tokens=max_new_tokens,
                ignore_eos=ignore_eos,
                timeout_sec=timeout_sec,
                enable_lora=enable_lora,
                lora_map=lora_map,
                run_tag=f"warmup_{mode}_{i}",
            )
        rows: list[dict[str, Any]] = []
        for i in range(repeats):
            rows.append(
                _run_round(
                    engine=engine,
                    reqs=reqs,
                    max_new_tokens=max_new_tokens,
                    ignore_eos=ignore_eos,
                    timeout_sec=timeout_sec,
                    enable_lora=enable_lora,
                    lora_map=lora_map,
                    run_tag=f"repeat_{mode}_{i}",
                )
            )
        return rows
    finally:
        _cleanup_engine(engine)


def _common_series_kwargs(
    args: argparse.Namespace,
    *,
    model_name: str,
    model_path: str,
    reqs: list[Req],
    enable_lora: bool,
    mode: str,
    enable_chunked_prefill: bool,
    adapter_a: Optional[str] = None,
    adapter_b: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "model_name": model_name,
        "reqs": reqs,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
        "timeout_sec": args.timeout_sec,
        "mode": mode,
        "enable_lora": enable_lora,
        "warmup_iters": args.warmup_iters,
        "repeats": args.repeats,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "queue_reorder_mode": args.queue_reorder_mode,
        "queue_reorder_aging_quantum_us": args.queue_reorder_aging_quantum_us,
        "phase1_objective_mode": args.phase1_objective_mode,
        "phase1_gamma": args.phase1_gamma,
        "phase1_ingress_target_chunk": args.phase1_ingress_target_chunk,
        "phase1_ingress_direct_authoritative": args.phase1_ingress_direct_authoritative,
        "phase1_ingress_exact_chunk": args.phase1_ingress_exact_chunk,
        "phase1_force_min_chunk": args.phase1_force_min_chunk,
        "phase1_target_long_fraction": args.phase1_target_long_fraction,
        "phase1_runtime_adaptive_enabled": args.phase1_runtime_adaptive_enabled,
        "phase1_runtime_aggressive_long_fraction": args.phase1_runtime_aggressive_long_fraction,
        "phase1_runtime_conservative_long_fraction": args.phase1_runtime_conservative_long_fraction,
        "phase1_runtime_aggressive_ingress_target_chunk": args.phase1_runtime_aggressive_ingress_target_chunk,
        "phase1_runtime_conservative_ingress_target_chunk": args.phase1_runtime_conservative_ingress_target_chunk,
        "phase1_runtime_queue_high_watermark": args.phase1_runtime_queue_high_watermark,
        "phase1_runtime_waiting_short_high_watermark": args.phase1_runtime_waiting_short_high_watermark,
        "phase1_runtime_wait_us_high_watermark": args.phase1_runtime_wait_us_high_watermark,
        "phase1_runtime_long_high_watermark": args.phase1_runtime_long_high_watermark,
        "phase1_runtime_urgency_discount": args.phase1_runtime_urgency_discount,
        "phase1_runtime_ema_alpha": args.phase1_runtime_ema_alpha,
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
        "phase2_enable_mixed_prefill_decode": args.phase2_enable_mixed_prefill_decode,
        "phase2_min_hetero_ratio": args.phase2_min_hetero_ratio,
        "phase2_min_long_prefill": args.phase2_min_long_prefill,
        "phase2_min_pressure_ratio": args.phase2_min_pressure_ratio,
        "phase2_enable_scheduler_cashout": args.phase2_enable_scheduler_cashout,
        "phase2_enable_execution_escape": args.phase2_enable_execution_escape,
        "phase2_execution_escape_mode": args.phase2_execution_escape_mode,
        "phase2_execution_escape_spillover_cap": args.phase2_execution_escape_spillover_cap,
        "phase2_execution_escape_max_active": args.phase2_execution_escape_max_active,
        "phase2_runtime_adaptive_enabled": args.phase2_runtime_adaptive_enabled,
        "phase2_runtime_low_pressure_min_hetero_ratio": args.phase2_runtime_low_pressure_min_hetero_ratio,
        "phase2_runtime_high_pressure_min_hetero_ratio": args.phase2_runtime_high_pressure_min_hetero_ratio,
        "phase2_runtime_low_pressure_min_pressure_ratio": args.phase2_runtime_low_pressure_min_pressure_ratio,
        "phase2_runtime_high_pressure_min_pressure_ratio": args.phase2_runtime_high_pressure_min_pressure_ratio,
        "phase2_runtime_low_pressure_min_long_prefill": args.phase2_runtime_low_pressure_min_long_prefill,
        "phase2_runtime_high_pressure_min_long_prefill": args.phase2_runtime_high_pressure_min_long_prefill,
        "phase2_runtime_low_pressure_escape_spillover_cap": args.phase2_runtime_low_pressure_escape_spillover_cap,
        "phase2_runtime_high_pressure_escape_spillover_cap": args.phase2_runtime_high_pressure_escape_spillover_cap,
        "phase2_runtime_low_pressure_escape_max_active": args.phase2_runtime_low_pressure_escape_max_active,
        "phase2_runtime_high_pressure_escape_max_active": args.phase2_runtime_high_pressure_escape_max_active,
        "phase2_runtime_disable_execution_escape_below_pressure": args.phase2_runtime_disable_execution_escape_below_pressure,
        "max_num_partial_prefills": args.max_num_partial_prefills,
        "max_long_partial_prefills": args.max_long_partial_prefills,
        "enable_chunked_prefill": enable_chunked_prefill,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "phase2_dispatch_mode": args.phase2_dispatch_mode,
        "trust_remote_code": args.trust_remote_code,
    }


def _run_series(
    args: argparse.Namespace,
    *,
    model_name: str,
    model_path: str,
    reqs: list[Req],
    enable_lora: bool,
    mode: str,
    enable_chunked_prefill: bool,
    adapter_a: Optional[str] = None,
    adapter_b: Optional[str] = None,
) -> list[dict[str, Any]]:
    return _run_mode_series(
        **_common_series_kwargs(
            args,
            model_name=model_name,
            model_path=model_path,
            reqs=reqs,
            enable_lora=enable_lora,
            mode=mode,
            enable_chunked_prefill=enable_chunked_prefill,
            adapter_a=adapter_a,
            adapter_b=adapter_b,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeated Wave-Slice evaluation with warmup/repeats and error rates.",
    )
    parser.add_argument(
        "--model-path",
        default="mistralai/Mistral-7B-v0.1",
    )
    parser.add_argument("--model-name", default="Mistral-7B-v0.1")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force each request to decode up to --max-new-tokens by ignoring EOS.",
    )
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--short-repeat", type=int, default=16)
    parser.add_argument("--short-a-repeat", type=int, default=None)
    parser.add_argument("--short-b-repeat", type=int, default=None)
    parser.add_argument("--long-repeat", type=int, default=320)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1536)
    parser.add_argument("--max-num-partial-prefills", type=int, default=1)
    parser.add_argument("--max-long-partial-prefills", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument(
        "--queue-reorder-mode",
        choices=["sjf", "hrrn", "aging"],
        default="sjf",
        help="Queue reorder policy used when Phase-I queue reordering is enabled.",
    )
    parser.add_argument(
        "--queue-reorder-aging-quantum-us",
        type=float,
        default=20_000.0,
        help="Aging quantum in microseconds when --queue-reorder-mode=aging.",
    )
    parser.add_argument(
        "--phase1-objective-mode",
        choices=["fair_escape", "pure_gain"],
        default="fair_escape",
    )
    parser.add_argument(
        "--phase1-baseline-mode",
        choices=["chunked", "no_chunk", "both"],
        default="both",
        help="Phase-I controls: chunked baseline only, no-chunk baseline only, or both.",
    )
    parser.add_argument(
        "--phase1-ingress-target-chunk",
        type=int,
        default=384,
        help="Authoritative ingress target chunk for Phase-I exact-chunk experiments.",
    )
    parser.add_argument(
        "--phase1-gamma",
        type=float,
        default=2.0,
        help="Penalty amplification gamma for Phase-I scheduler.",
    )
    parser.add_argument(
        "--phase1-ingress-direct-authoritative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ingress direct authoritative chunk override for Phase-I.",
    )
    parser.add_argument(
        "--phase1-ingress-exact-chunk",
        action="store_true",
        default=False,
        help="When authoritative ingress is enabled, use the exact target chunk instead of bucket-down mapping.",
    )
    parser.add_argument(
        "--phase1-force-min-chunk",
        type=int,
        default=128,
        help="Minimum aggressive Phase-I chunk used by the fairness/LoRA ingress heuristics.",
    )
    parser.add_argument(
        "--phase1-target-long-fraction",
        type=float,
        default=0.33,
        help="Fraction of a long prefill used by the LoRA-aware ingress cap heuristic.",
    )
    parser.add_argument(
        "--phase1-runtime-adaptive-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adapt Phase-I chunk targets from live scheduler queue pressure.",
    )
    parser.add_argument("--phase1-runtime-aggressive-long-fraction", type=float, default=0.33)
    parser.add_argument("--phase1-runtime-conservative-long-fraction", type=float, default=0.50)
    parser.add_argument("--phase1-runtime-aggressive-ingress-target-chunk", type=int, default=768)
    parser.add_argument("--phase1-runtime-conservative-ingress-target-chunk", type=int, default=1536)
    parser.add_argument("--phase1-runtime-queue-high-watermark", type=int, default=8)
    parser.add_argument("--phase1-runtime-waiting-short-high-watermark", type=int, default=4)
    parser.add_argument("--phase1-runtime-wait-us-high-watermark", type=float, default=1_000_000.0)
    parser.add_argument("--phase1-runtime-long-high-watermark", type=int, default=3072)
    parser.add_argument("--phase1-runtime-urgency-discount", type=float, default=0.55)
    parser.add_argument("--phase1-runtime-ema-alpha", type=float, default=0.35)
    parser.add_argument(
        "--phase12-phase2-gate-mode",
        choices=["hard", "soft"],
        default="soft",
        help="Joint coordination gate mode for Phase-II when Phase-I and Phase-II are both enabled.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-ratio-scale",
        type=float,
        default=1.15,
        help="Multiplier on heterogeneity ratio threshold for the soft joint Phase-II gate.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-pressure-scale",
        type=float,
        default=1.10,
        help="Multiplier on pressure ratio threshold for the soft joint Phase-II gate.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-min-long-prefill",
        type=int,
        default=512,
        help="Minimum long-prefill length that can unlock the soft joint Phase-II gate without recent Phase-I activity.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-allow-mixed-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether mixed prefill/decode pressure can unlock the soft joint Phase-II gate.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-recent-strength-floor",
        type=float,
        default=0.08,
        help="Minimum decayed recent Phase-I strength that counts as a cash-out signal for the soft joint gate.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-require-cashout-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a concrete cash-out signal (recent Phase-I strength or mixed decode pressure) before soft strong-prefill Phase-II dispatch.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-recent-chunk-match-scale",
        type=float,
        default=1.5,
        help="Allow soft Phase-II cash-out only when the current smallest prefill still matches the recent Phase-I chunk window up to this scale.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-window-score-threshold",
        type=float,
        default=0.95,
        help="Minimum joint window-quality score needed before soft Phase-II dispatch is treated as a reliable cash-out.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-window-recent-weight",
        type=float,
        default=0.40,
        help="Weight of recent Phase-I strength in the soft joint window-quality score.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-window-chunk-weight",
        type=float,
        default=0.25,
        help="Weight of chunk-window match quality in the soft joint window-quality score.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-window-pressure-weight",
        type=float,
        default=0.20,
        help="Weight of current Phase-II pressure in the soft joint window-quality score.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-window-ratio-weight",
        type=float,
        default=0.10,
        help="Weight of current heterogeneity ratio in the soft joint window-quality score.",
    )
    parser.add_argument(
        "--phase12-phase2-soft-window-decode-bonus",
        type=float,
        default=0.10,
        help="Bonus added to the soft joint window-quality score when mixed prefill/decode pressure is present.",
    )
    parser.add_argument("--include-strict", action="store_true")
    parser.add_argument(
        "--include-phase12",
        action="store_true",
        help="Also run the combined Phase-I + Phase-II LoRA series.",
    )
    parser.add_argument(
        "--include-phase1-lora-only",
        action="store_true",
        help="Also run the Phase-I-only scheduler on the LoRA workload for like-for-like comparison against the LoRA baseline.",
    )
    parser.add_argument(
        "--phase2-dispatch-mode",
        choices=["synchronized", "async_experimental"],
        default="synchronized",
    )
    parser.add_argument(
        "--phase2-enable-mixed-prefill-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--phase2-min-hetero-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--phase2-min-long-prefill",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--phase2-min-pressure-ratio",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--phase2-enable-scheduler-cashout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the scheduler cashout/prehide path that mirrors the v0 Phase-II behavior.",
    )
    parser.add_argument(
        "--phase2-enable-execution-escape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the v1 execution-escape bookkeeping path.",
    )
    parser.add_argument(
        "--phase2-execution-escape-mode",
        choices=["broad_partition", "beneficiary_only", "bounded_spillover"],
        default="bounded_spillover",
    )
    parser.add_argument(
        "--phase2-execution-escape-spillover-cap",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--phase2-execution-escape-max-active",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--phase2-runtime-adaptive-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adapt Phase-II gates from live scheduler pressure when Phase-I runtime adaptation is active.",
    )
    parser.add_argument("--phase2-runtime-low-pressure-min-hetero-ratio", type=float, default=6.0)
    parser.add_argument("--phase2-runtime-high-pressure-min-hetero-ratio", type=float, default=4.0)
    parser.add_argument("--phase2-runtime-low-pressure-min-pressure-ratio", type=float, default=6.0)
    parser.add_argument("--phase2-runtime-high-pressure-min-pressure-ratio", type=float, default=4.0)
    parser.add_argument("--phase2-runtime-low-pressure-min-long-prefill", type=int, default=1024)
    parser.add_argument("--phase2-runtime-high-pressure-min-long-prefill", type=int, default=768)
    parser.add_argument("--phase2-runtime-low-pressure-escape-spillover-cap", type=int, default=1)
    parser.add_argument("--phase2-runtime-high-pressure-escape-spillover-cap", type=int, default=3)
    parser.add_argument("--phase2-runtime-low-pressure-escape-max-active", type=int, default=2)
    parser.add_argument("--phase2-runtime-high-pressure-escape-max-active", type=int, default=5)
    parser.add_argument("--phase2-runtime-disable-execution-escape-below-pressure", type=float, default=-1.0)
    parser.add_argument(
        "--phase2-baseline-enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the paired Phase-II baseline uses chunked prefill. Disable this to capture vLLM default continuous batching on the LoRA phase2 segment.",
    )
    parser.add_argument(
        "--adapter-a",
        default="",
    )
    parser.add_argument(
        "--adapter-b",
        default="",
    )
    parser.add_argument(
        "--adapters-root",
        default=os.path.join("results", "synthetic_adapters"),
        help="Directory used when synthetic adapters need to be auto-created.",
    )
    parser.add_argument(
        "--no-auto-build-adapters",
        action="store_true",
        help="Disable automatic synthetic adapter creation when adapter paths are omitted.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output json path. Default: results/waveslice_repeated_eval_<ts>.json",
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Run Phase-I only and skip LoRA Phase-II blocks.",
    )
    parser.add_argument(
        "--requests-json",
        default="",
        help="Optional JSON file containing non-LoRA requests.",
    )
    parser.add_argument(
        "--lora-requests-json",
        default="",
        help="Optional JSON file containing LoRA requests.",
    )
    parser.add_argument(
        "--serialize-gpu-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serialize GPU-backed experiments through a global file lock.",
    )
    parser.add_argument(
        "--gpu-lock-path",
        default="",
        help="Optional file path used for the global GPU experiment lock.",
    )
    args = parser.parse_args()

    short_a_repeat = args.short_a_repeat if args.short_a_repeat is not None else args.short_repeat
    short_b_repeat = args.short_b_repeat if args.short_b_repeat is not None else args.short_repeat

    max_prompt_tokens = max(16, args.max_model_len - args.max_new_tokens - 16)
    if args.requests_json.strip():
        reqs = load_reqs_json(args.requests_json)
        tok_lens = measure_input_tokens(
            args.model_path,
            reqs,
            trust_remote_code=args.trust_remote_code,
        )
        fitted_long_repeat = args.long_repeat
    else:
        reqs, tok_lens, fitted_long_repeat = fit_requests_to_context(
            model_path=args.model_path,
            short_a_repeat=short_a_repeat,
            short_b_repeat=short_b_repeat,
            long_repeat=args.long_repeat,
            max_prompt_tokens=max_prompt_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    if args.lora_requests_json.strip():
        lora_reqs = load_reqs_json(args.lora_requests_json)
        lora_tok_lens = measure_input_tokens(
            args.model_path,
            lora_reqs,
            trust_remote_code=args.trust_remote_code,
        )
        fitted_lora_long_repeat = args.long_repeat
    else:
        lora_reqs, lora_tok_lens, fitted_lora_long_repeat = fit_lora_requests_to_context(
            model_path=args.model_path,
            short_a_repeat=short_a_repeat,
            short_b_repeat=short_b_repeat,
            long_repeat=args.long_repeat,
            max_prompt_tokens=max_prompt_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    short_lens = [v for k, v in tok_lens.items() if "short" in k]
    long_lens = [v for k, v in tok_lens.items() if "long" in k]

    def _fmt_range(vals: list[int]) -> str:
        if not vals:
            return "n/a"
        return f"[{min(vals)}, {max(vals)}]"

    print("[Eval] Request token lengths")
    print(f"  per_request={tok_lens}")
    print(f"  short_range={_fmt_range(short_lens)}, long_range={_fmt_range(long_lens)}")
    print(f"  decode_max_new_tokens={args.max_new_tokens}")
    if fitted_long_repeat != args.long_repeat:
        print(f"  long_repeat auto-adjusted: {args.long_repeat} -> {fitted_long_repeat}")
    print("[Eval] LoRA request token lengths")
    print(f"  per_request={lora_tok_lens}")
    if fitted_lora_long_repeat != args.long_repeat:
        print(f"  lora_long_repeat auto-adjusted: {args.long_repeat} -> {fitted_lora_long_repeat}")

    need_lora_adapters = (
        (not args.skip_phase2)
        or args.include_phase12
        or args.include_strict
        or args.include_phase1_lora_only
    )
    if need_lora_adapters:
        adapter_a = args.adapter_a.strip()
        adapter_b = args.adapter_b.strip()
        if not adapter_a or not adapter_b:
            if args.no_auto_build_adapters:
                print("[Eval] adapter paths were not provided and auto-build is disabled.")
                return 1
            print("[Eval] auto-building synthetic adapters")
            adapter_a, adapter_b = _ensure_eval_adapters(
                model_path=args.model_path,
                adapters_root=args.adapters_root,
                trust_remote_code=args.trust_remote_code,
            )
        elif not (os.path.exists(adapter_a) and os.path.exists(adapter_b)):
            if args.no_auto_build_adapters:
                print("[Eval] adapters not found; cannot run Phase-II LoRA repeated test.")
                print(f"  expected A={adapter_a}")
                print(f"  expected B={adapter_b}")
                return 1
            print("[Eval] adapter paths missing on disk; auto-building synthetic adapters")
            adapter_a, adapter_b = _ensure_eval_adapters(
                model_path=args.model_path,
                adapters_root=args.adapters_root,
                trust_remote_code=args.trust_remote_code,
            )
        args.adapter_a = adapter_a
        args.adapter_b = adapter_b

    print(f"[Eval] warmup_iters={args.warmup_iters}, repeats={args.repeats}")

    need_chunked_baseline = args.phase1_baseline_mode in {"chunked", "both"}
    need_no_chunk_baseline = args.phase1_baseline_mode in {"no_chunk", "both"}

    phase1_base_rounds: list[dict[str, Any]] = []
    phase1_base_repeat_rounds: list[dict[str, Any]] = []
    if need_chunked_baseline:
        print("[Eval] Running Phase-I chunked baseline series")
        phase1_base_rounds = _run_series(
            args,
            model_name=args.model_name,
            model_path=args.model_path,
            reqs=reqs,
            enable_lora=False,
            mode="baseline",
            enable_chunked_prefill=True,
        )
        print("[Eval] Running Phase-I chunked baseline noise series")
        phase1_base_repeat_rounds = _run_series(
            args,
            model_name=args.model_name,
            model_path=args.model_path,
            reqs=reqs,
            enable_lora=False,
            mode="baseline",
            enable_chunked_prefill=True,
        )

    phase1_no_chunk_rounds: list[dict[str, Any]] = []
    phase1_no_chunk_repeat_rounds: list[dict[str, Any]] = []
    if need_no_chunk_baseline:
        print("[Eval] Running Phase-I no-chunk baseline series")
        phase1_no_chunk_rounds = _run_series(
            args,
            model_name=args.model_name,
            model_path=args.model_path,
            reqs=reqs,
            enable_lora=False,
            mode="baseline",
            enable_chunked_prefill=False,
        )
        print("[Eval] Running Phase-I no-chunk baseline noise series")
        phase1_no_chunk_repeat_rounds = _run_series(
            args,
            model_name=args.model_name,
            model_path=args.model_path,
            reqs=reqs,
            enable_lora=False,
            mode="baseline",
            enable_chunked_prefill=False,
        )

    if not phase1_base_rounds:
        phase1_base_rounds = phase1_no_chunk_rounds
        phase1_base_repeat_rounds = phase1_no_chunk_repeat_rounds

    print("[Eval] Running Phase-I Wave-Slice series")
    phase1_wave_rounds = _run_series(
        args,
        model_name=args.model_name,
        model_path=args.model_path,
        reqs=reqs,
        enable_lora=False,
        mode="phase1_only",
        enable_chunked_prefill=True,
    )
    phase1 = run_phase1_pair(
        base_rows=phase1_base_rounds,
        base_repeat_rows=phase1_base_repeat_rounds,
        wave_rows=phase1_wave_rounds,
    )
    phase1_no_chunk_control = None
    if need_chunked_baseline and need_no_chunk_baseline:
        phase1_no_chunk_control = run_phase1_pair(
            base_rows=phase1_no_chunk_rounds,
            base_repeat_rows=phase1_no_chunk_repeat_rounds,
            wave_rows=phase1_base_rounds,
        )

    for i, row in enumerate(phase1["rows"], start=1):
        print(
            f"[Repeat] {i}/{args.repeats} Phase-I "
            f"ttft_gain={row.get('ttft_improve_ratio')} "
            f"wall_gain={row.get('round_wall_improve_ratio')} "
            f"error_rate={row.get('error_rate')} "
            f"base_noise_err={row.get('baseline_noise_error_rate')} "
            f"apply_ratio={row.get('scheduler_apply_ratio')}"
        )

    if args.skip_phase2:
        ts = int(time.time())
        result = {
            "phase1": phase1["summary"],
            "phase1_rows": phase1["rows"],
            "phase1_chunked_vs_no_chunk": (
                phase1_no_chunk_control["summary"] if phase1_no_chunk_control is not None else None
            ),
            "phase1_chunked_vs_no_chunk_rows": (
                phase1_no_chunk_control["rows"] if phase1_no_chunk_control is not None else None
            ),
            "request_token_lengths": tok_lens,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "phase1_objective_mode": args.phase1_objective_mode,
            "phase1_baseline_mode": args.phase1_baseline_mode,
        }
        out_path = args.out_json or os.path.join(
            "results",
            f"waveslice_phase1_only_eval_{ts}.json",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("\n[Summary] Phase-I")
        for key, stats in phase1["summary"].items():
            print(f"  {key}: {stats}")
        if phase1_no_chunk_control is not None:
            print("\n[Summary] Chunked-vs-NoChunk")
            for key, stats in phase1_no_chunk_control["summary"].items():
                print(f"  {key}: {stats}")
        print(f"\n[Saved] {out_path}")
        return 0

    print("[Eval] Running Phase-II baseline series A")
    phase2_base_rounds = _run_series(
        args,
        model_name=args.model_name,
        model_path=args.model_path,
        reqs=lora_reqs,
        enable_lora=True,
        mode="baseline_lora_compat",
        enable_chunked_prefill=bool(args.phase2_baseline_enable_chunked_prefill),
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
    )
    print("[Eval] Running Phase-II baseline series B (noise floor)")
    phase2_base_repeat_rounds = _run_series(
        args,
        model_name=args.model_name,
        model_path=args.model_path,
        reqs=lora_reqs,
        enable_lora=True,
        mode="baseline_lora_compat",
        enable_chunked_prefill=bool(args.phase2_baseline_enable_chunked_prefill),
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
    )
    print("[Eval] Running Phase-II Wave-Slice series")
    phase2_wave_rounds = _run_series(
        args,
        model_name=args.model_name,
        model_path=args.model_path,
        reqs=lora_reqs,
        enable_lora=True,
        mode="phase2_lora",
        enable_chunked_prefill=True,
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
    )
    phase2_strict_rounds: Optional[list[dict[str, Any]]] = None
    if args.include_strict:
        print("[Eval] Running Phase-II strict Wave-Slice series")
        phase2_strict_rounds = _run_series(
            args,
            model_name=args.model_name,
            model_path=args.model_path,
            reqs=lora_reqs,
            enable_lora=True,
            mode="phase2_lora_strict",
            enable_chunked_prefill=True,
            adapter_a=args.adapter_a,
            adapter_b=args.adapter_b,
        )
    phase2 = run_phase2_block(
        base_rows=phase2_base_rounds,
        base_repeat_rows=phase2_base_repeat_rounds,
        wave_rows=phase2_wave_rounds,
        strict_rows=phase2_strict_rounds,
        include_strict=args.include_strict,
    )
    phase1_lora: Optional[dict[str, Any]] = None
    if args.include_phase1_lora_only:
        print("[Eval] Running Phase-I-only LoRA Wave-Slice series")
        phase1_lora_rounds = _run_series(
            args,
            model_name=args.model_name,
            model_path=args.model_path,
            reqs=lora_reqs,
            enable_lora=True,
            mode="phase1_lora_only",
            enable_chunked_prefill=True,
            adapter_a=args.adapter_a,
            adapter_b=args.adapter_b,
        )
        phase1_lora = run_phase2_block(
            base_rows=phase2_base_rounds,
            base_repeat_rows=phase2_base_repeat_rounds,
            wave_rows=phase1_lora_rounds,
            strict_rows=None,
            include_strict=False,
        )

    phase12: Optional[dict[str, Any]] = None
    if args.include_phase12:
        print("[Eval] Running Phase-I + Phase-II Wave-Slice series")
        phase12_rounds = _run_series(
            args,
            model_name=args.model_name,
            model_path=args.model_path,
            reqs=lora_reqs,
            enable_lora=True,
            mode="phase12_lora",
            enable_chunked_prefill=True,
            adapter_a=args.adapter_a,
            adapter_b=args.adapter_b,
        )
        phase12 = run_phase2_block(
            base_rows=phase2_base_rounds,
            base_repeat_rows=phase2_base_repeat_rounds,
            wave_rows=phase12_rounds,
            strict_rows=None,
            include_strict=False,
        )

    for i, p2 in enumerate(phase2["rows"], start=1):
        print(
            f"[Repeat] {i}/{args.repeats} Phase-II "
            f"ttft_gain={p2.get('ttft_improve_ratio')} "
            f"slow_gain={p2.get('slowdown_improve_ratio')} "
            f"wall_gain={p2.get('round_wall_improve_ratio')} "
            f"err={p2.get('wave_error_rate')} "
            f"base_noise_err={p2.get('baseline_noise_error_rate')} "
            f"apply_ratio={p2.get('phase2_apply_ratio')}"
        )

    summary = build_summary(
        args=args,
        short_a_repeat=short_a_repeat,
        short_b_repeat=short_b_repeat,
        tok_lens=tok_lens,
        need_chunked_baseline=need_chunked_baseline,
        need_no_chunk_baseline=need_no_chunk_baseline,
        phase1_base_rounds=phase1_base_rounds,
        phase1_no_chunk_rounds=phase1_no_chunk_rounds,
        phase1=phase1,
        phase2=phase2,
        phase1_no_chunk_control=phase1_no_chunk_control,
        phase1_lora=phase1_lora,
        phase12=phase12,
    )

    print_summary(
        summary,
        include_strict=args.include_strict,
        include_phase1_lora_only=args.include_phase1_lora_only,
        include_phase12=args.include_phase12,
    )

    out_json = write_summary_json(summary, args.out_json)
    print(f"\n[Output] {out_json}")
    return 0


if __name__ == "__main__":
    serialize_gpu_tests = bool_arg_from_argv("serialize-gpu-tests", True)
    gpu_lock_path = str_arg_from_argv("gpu-lock-path", "")
    model_name = str_arg_from_argv("model-name", "unknown-model")
    with gpu_experiment_lock(
        label=f"evaluate:{model_name}",
        enabled=serialize_gpu_tests,
        lock_path=gpu_lock_path or None,
    ):
        raise SystemExit(main())
