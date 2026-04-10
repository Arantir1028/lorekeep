from __future__ import annotations

import argparse
import json
import os
from typing import Any

from engine.runtime_bootstrap import bootstrap_vllm_runtime

bootstrap_vllm_runtime()

from tests.evaluate_waveslice_claims import (
    _load_reqs_json,
    _measure_input_tokens,
    _run_mode_series,
)


def _beneficiary_req_ids(reqs: list[Any]) -> list[str]:
    ordered = sorted(
        reqs,
        key=lambda r: (float(getattr(r, "arrival_offset_s", 0.0) or 0.0), str(r.req_id)),
    )
    first_long_seen = False
    ids: list[str] = []
    for req in ordered:
        if not bool(getattr(req, "is_short", False)):
            first_long_seen = True
            continue
        if first_long_seen:
            ids.append(str(req.req_id))
    return ids


def _subset_stats(request_timings: dict[str, Any], req_ids: list[str]) -> dict[str, Any]:
    rows = [request_timings[rid] for rid in req_ids if rid in request_timings]
    def _mean(key: str) -> float | None:
        vals = [float(v[key]) for v in rows if v.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None
    return {
        "count": len(rows),
        "first_latency_ms_avg": _mean("first_latency_ms"),
        "finish_latency_ms_avg": _mean("finish_latency_ms"),
        "first_event_index_avg": _mean("first_event_index"),
        "finish_event_index_avg": _mean("finish_event_index"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight beneficiary escape-lane observer.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--requests-json", required=True)
    parser.add_argument("--lora-requests-json", required=True)
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--queue-reorder-mode", default="sjf")
    parser.add_argument("--queue-reorder-aging-quantum-us", type=float, default=20000.0)
    parser.add_argument("--phase1-objective-mode", default="fair_escape")
    parser.add_argument("--phase1-gamma", type=float, default=2.0)
    parser.add_argument("--phase1-ingress-target-chunk", type=int, default=384)
    parser.add_argument(
        "--phase1-ingress-direct-authoritative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--phase1-ingress-exact-chunk", action="store_true", default=False)
    parser.add_argument("--phase12-phase2-gate-mode", default="soft")
    parser.add_argument("--phase12-phase2-soft-ratio-scale", type=float, default=1.15)
    parser.add_argument("--phase12-phase2-soft-pressure-scale", type=float, default=1.10)
    parser.add_argument("--phase12-phase2-soft-min-long-prefill", type=int, default=512)
    parser.add_argument("--phase12-phase2-soft-allow-mixed-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase12-phase2-soft-recent-strength-floor", type=float, default=0.08)
    parser.add_argument("--phase12-phase2-soft-require-cashout-signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase12-phase2-soft-recent-chunk-match-scale", type=float, default=1.5)
    parser.add_argument("--phase12-phase2-soft-window-score-threshold", type=float, default=0.95)
    parser.add_argument("--phase12-phase2-soft-window-recent-weight", type=float, default=0.40)
    parser.add_argument("--phase12-phase2-soft-window-chunk-weight", type=float, default=0.25)
    parser.add_argument("--phase12-phase2-soft-window-pressure-weight", type=float, default=0.20)
    parser.add_argument("--phase12-phase2-soft-window-ratio-weight", type=float, default=0.10)
    parser.add_argument("--phase12-phase2-soft-window-decode-bonus", type=float, default=0.10)
    parser.add_argument("--phase12-phase2-scheduler-cashout-soft-floor", type=float, default=0.55)
    parser.add_argument("--phase12-phase2-scheduler-cashout-quality-floor", type=float, default=0.78)
    parser.add_argument("--phase12-phase2-scheduler-cashout-cooldown-ticks", type=int, default=2)
    parser.add_argument("--phase2-execution-escape-mode", default="bounded_spillover")
    parser.add_argument("--phase2-execution-escape-spillover-cap", type=int, default=3)
    parser.add_argument("--phase2-execution-escape-max-active", type=int, default=5)
    parser.add_argument("--phase2-dispatch-mode", default="synchronized")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    reqs = _load_reqs_json(args.requests_json)
    lora_reqs = _load_reqs_json(args.lora_requests_json)
    lora_tok_lens = _measure_input_tokens(
        args.model_path,
        lora_reqs,
        trust_remote_code=args.trust_remote_code,
    )
    beneficiary_ids = _beneficiary_req_ids(lora_reqs)

    print("[Observe] Running LoRA baseline series")
    base_rows = _run_mode_series(
        model_path=args.model_path,
        model_name=args.model_name,
        reqs=lora_reqs,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
        mode="baseline_lora_compat",
        enable_lora=True,
        warmup_iters=0,
        repeats=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        queue_reorder_mode=args.queue_reorder_mode,
        queue_reorder_aging_quantum_us=args.queue_reorder_aging_quantum_us,
        phase1_objective_mode=args.phase1_objective_mode,
        phase1_gamma=args.phase1_gamma,
        phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
        phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
        phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
        phase12_phase2_gate_mode=args.phase12_phase2_gate_mode,
        phase12_phase2_soft_ratio_scale=args.phase12_phase2_soft_ratio_scale,
        phase12_phase2_soft_pressure_scale=args.phase12_phase2_soft_pressure_scale,
        phase12_phase2_soft_min_long_prefill=args.phase12_phase2_soft_min_long_prefill,
        phase12_phase2_soft_allow_mixed_decode=args.phase12_phase2_soft_allow_mixed_decode,
        phase12_phase2_soft_recent_strength_floor=args.phase12_phase2_soft_recent_strength_floor,
        phase12_phase2_soft_require_cashout_signal=args.phase12_phase2_soft_require_cashout_signal,
        phase12_phase2_soft_recent_chunk_match_scale=args.phase12_phase2_soft_recent_chunk_match_scale,
        phase12_phase2_soft_window_score_threshold=args.phase12_phase2_soft_window_score_threshold,
        phase12_phase2_soft_window_recent_weight=args.phase12_phase2_soft_window_recent_weight,
        phase12_phase2_soft_window_chunk_weight=args.phase12_phase2_soft_window_chunk_weight,
        phase12_phase2_soft_window_pressure_weight=args.phase12_phase2_soft_window_pressure_weight,
        phase12_phase2_soft_window_ratio_weight=args.phase12_phase2_soft_window_ratio_weight,
        phase12_phase2_soft_window_decode_bonus=args.phase12_phase2_soft_window_decode_bonus,
        phase12_phase2_scheduler_cashout_soft_floor=args.phase12_phase2_scheduler_cashout_soft_floor,
        phase12_phase2_scheduler_cashout_quality_floor=args.phase12_phase2_scheduler_cashout_quality_floor,
        phase12_phase2_scheduler_cashout_cooldown_ticks=args.phase12_phase2_scheduler_cashout_cooldown_ticks,
        phase2_execution_escape_mode=args.phase2_execution_escape_mode,
        phase2_execution_escape_spillover_cap=args.phase2_execution_escape_spillover_cap,
        phase2_execution_escape_max_active=args.phase2_execution_escape_max_active,
        enable_chunked_prefill=True,
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
        phase2_dispatch_mode=args.phase2_dispatch_mode,
        trust_remote_code=args.trust_remote_code,
    )
    print("[Observe] Running Phase-I + Phase-II series")
    phase12_rows = _run_mode_series(
        model_path=args.model_path,
        model_name=args.model_name,
        reqs=lora_reqs,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
        mode="phase12_lora",
        enable_lora=True,
        warmup_iters=0,
        repeats=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        queue_reorder_mode=args.queue_reorder_mode,
        queue_reorder_aging_quantum_us=args.queue_reorder_aging_quantum_us,
        phase1_objective_mode=args.phase1_objective_mode,
        phase1_gamma=args.phase1_gamma,
        phase1_ingress_target_chunk=args.phase1_ingress_target_chunk,
        phase1_ingress_direct_authoritative=args.phase1_ingress_direct_authoritative,
        phase1_ingress_exact_chunk=args.phase1_ingress_exact_chunk,
        phase12_phase2_gate_mode=args.phase12_phase2_gate_mode,
        phase12_phase2_soft_ratio_scale=args.phase12_phase2_soft_ratio_scale,
        phase12_phase2_soft_pressure_scale=args.phase12_phase2_soft_pressure_scale,
        phase12_phase2_soft_min_long_prefill=args.phase12_phase2_soft_min_long_prefill,
        phase12_phase2_soft_allow_mixed_decode=args.phase12_phase2_soft_allow_mixed_decode,
        phase12_phase2_soft_recent_strength_floor=args.phase12_phase2_soft_recent_strength_floor,
        phase12_phase2_soft_require_cashout_signal=args.phase12_phase2_soft_require_cashout_signal,
        phase12_phase2_soft_recent_chunk_match_scale=args.phase12_phase2_soft_recent_chunk_match_scale,
        phase12_phase2_soft_window_score_threshold=args.phase12_phase2_soft_window_score_threshold,
        phase12_phase2_soft_window_recent_weight=args.phase12_phase2_soft_window_recent_weight,
        phase12_phase2_soft_window_chunk_weight=args.phase12_phase2_soft_window_chunk_weight,
        phase12_phase2_soft_window_pressure_weight=args.phase12_phase2_soft_window_pressure_weight,
        phase12_phase2_soft_window_ratio_weight=args.phase12_phase2_soft_window_ratio_weight,
        phase12_phase2_soft_window_decode_bonus=args.phase12_phase2_soft_window_decode_bonus,
        phase12_phase2_scheduler_cashout_soft_floor=args.phase12_phase2_scheduler_cashout_soft_floor,
        phase12_phase2_scheduler_cashout_quality_floor=args.phase12_phase2_scheduler_cashout_quality_floor,
        phase12_phase2_scheduler_cashout_cooldown_ticks=args.phase12_phase2_scheduler_cashout_cooldown_ticks,
        phase2_execution_escape_mode=args.phase2_execution_escape_mode,
        phase2_execution_escape_spillover_cap=args.phase2_execution_escape_spillover_cap,
        phase2_execution_escape_max_active=args.phase2_execution_escape_max_active,
        enable_chunked_prefill=True,
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
        phase2_dispatch_mode=args.phase2_dispatch_mode,
        trust_remote_code=args.trust_remote_code,
    )

    base = base_rows[0]
    wave = phase12_rows[0]
    base_beneficiary_stats = _subset_stats(base.get("request_timings") or {}, beneficiary_ids)
    wave_beneficiary_stats = _subset_stats(wave.get("request_timings") or {}, beneficiary_ids)

    first_latency_ratio = None
    if (
        base_beneficiary_stats.get("first_latency_ms_avg") is not None
        and wave_beneficiary_stats.get("first_latency_ms_avg") is not None
        and float(wave_beneficiary_stats["first_latency_ms_avg"]) > 0.0
    ):
        first_latency_ratio = (
            float(base_beneficiary_stats["first_latency_ms_avg"])
            / float(wave_beneficiary_stats["first_latency_ms_avg"])
        )

    finish_latency_ratio = None
    if (
        base_beneficiary_stats.get("finish_latency_ms_avg") is not None
        and wave_beneficiary_stats.get("finish_latency_ms_avg") is not None
        and float(wave_beneficiary_stats["finish_latency_ms_avg"]) > 0.0
    ):
        finish_latency_ratio = (
            float(base_beneficiary_stats["finish_latency_ms_avg"])
            / float(wave_beneficiary_stats["finish_latency_ms_avg"])
        )

    result = {
        "config": {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "requests_json": args.requests_json,
            "lora_requests_json": args.lora_requests_json,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "phase12_phase2_scheduler_cashout_soft_floor": args.phase12_phase2_scheduler_cashout_soft_floor,
            "phase12_phase2_scheduler_cashout_quality_floor": args.phase12_phase2_scheduler_cashout_quality_floor,
            "phase12_phase2_scheduler_cashout_cooldown_ticks": args.phase12_phase2_scheduler_cashout_cooldown_ticks,
            "phase2_execution_escape_mode": args.phase2_execution_escape_mode,
            "phase2_execution_escape_spillover_cap": args.phase2_execution_escape_spillover_cap,
            "phase2_execution_escape_max_active": args.phase2_execution_escape_max_active,
        },
        "token_lengths": lora_tok_lens,
        "beneficiary_req_ids": beneficiary_ids,
        "baseline": {
            "ttft_short_p99_ms": base.get("ttft_short_p99_ms"),
            "round_wall_ms": base.get("round_wall_ms"),
            "request_timings": base.get("request_timings"),
            "hook_report": base.get("hook_report"),
            "beneficiary_stats": base_beneficiary_stats,
        },
        "phase12": {
            "ttft_short_p99_ms": wave.get("ttft_short_p99_ms"),
            "round_wall_ms": wave.get("round_wall_ms"),
            "request_timings": wave.get("request_timings"),
            "hook_report": wave.get("hook_report"),
            "beneficiary_stats": wave_beneficiary_stats,
        },
        "beneficiary_deltas": {
            "first_latency_ms_avg_improve_ratio": first_latency_ratio,
            "finish_latency_ms_avg_improve_ratio": finish_latency_ratio,
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Output] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
