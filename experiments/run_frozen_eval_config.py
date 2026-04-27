from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.openworkload_support import load_config as _load_config


def _append_value(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    cmd.extend([f"--{flag}", text])


def _append_store_true(cmd: list[str], flag: str, value: Any) -> None:
    if bool(value):
        cmd.append(f"--{flag}")


def _append_bool_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.append(f"--{flag}" if bool(value) else f"--no-{flag}")


def build_eval_invocation(
    config: dict[str, Any],
    *,
    out_json_override: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    evaluator = str(config.get("evaluator") or "tests/evaluate_waveslice_claims.py")
    model_cfg = dict(config.get("model") or {})
    workload_cfg = dict(config.get("workload") or {})
    adapters_cfg = dict(config.get("adapters") or {})
    runtime_cfg = dict(config.get("runtime") or {})
    phase1_cfg = dict(config.get("phase1") or {})
    phase12_cfg = dict(config.get("phase12_soft_gate") or {})
    phase2_cfg = dict(config.get("phase2") or {})
    python_bin = str(runtime_cfg.get("python_bin") or sys.executable).strip()

    result_json = out_json_override or str(config.get("result_json") or "").strip()

    cmd: list[str] = [
        python_bin,
        evaluator,
    ]
    _append_value(cmd, "model-name", model_cfg.get("name"))
    _append_value(cmd, "model-path", model_cfg.get("path"))
    _append_value(cmd, "requests-json", workload_cfg.get("requests_json"))
    _append_value(cmd, "lora-requests-json", workload_cfg.get("lora_requests_json"))
    _append_value(cmd, "adapter-a", adapters_cfg.get("adapter_a"))
    _append_value(cmd, "adapter-b", adapters_cfg.get("adapter_b"))
    _append_store_true(cmd, "no-auto-build-adapters", True)
    include_phase12 = config.get("include_phase12")
    if include_phase12 is None:
        include_phase12 = True
    _append_store_true(cmd, "include-phase12", include_phase12)
    _append_store_true(cmd, "include-phase1-lora-only", config.get("include_phase1_lora_only"))
    _append_store_true(cmd, "include-strict", config.get("include_strict"))

    _append_value(cmd, "warmup-iters", runtime_cfg.get("warmup_iters"))
    _append_value(cmd, "repeats", runtime_cfg.get("repeats"))
    _append_value(cmd, "timeout-sec", runtime_cfg.get("timeout_sec"))
    _append_value(cmd, "max-new-tokens", runtime_cfg.get("max_new_tokens"))
    _append_value(cmd, "max-model-len", runtime_cfg.get("max_model_len"))
    _append_value(cmd, "max-num-batched-tokens", runtime_cfg.get("max_num_batched_tokens"))
    _append_value(cmd, "gpu-memory-utilization", runtime_cfg.get("gpu_memory_utilization"))
    _append_value(cmd, "queue-reorder-mode", runtime_cfg.get("queue_reorder_mode"))
    _append_value(cmd, "queue-reorder-aging-quantum-us", runtime_cfg.get("queue_reorder_aging_quantum_us"))
    _append_store_true(cmd, "trust-remote-code", runtime_cfg.get("trust_remote_code"))

    _append_value(cmd, "phase1-objective-mode", phase1_cfg.get("objective_mode"))
    _append_value(cmd, "phase1-baseline-mode", phase1_cfg.get("baseline_mode"))
    _append_value(cmd, "phase1-gamma", phase1_cfg.get("gamma"))
    _append_value(cmd, "phase1-ingress-target-chunk", phase1_cfg.get("ingress_target_chunk"))
    _append_bool_optional(cmd, "phase1-ingress-direct-authoritative", phase1_cfg.get("ingress_direct_authoritative"))
    _append_bool_optional(cmd, "phase1-ingress-exact-chunk", phase1_cfg.get("ingress_exact_chunk"))
    _append_value(cmd, "phase1-force-min-chunk", phase1_cfg.get("force_min_chunk"))
    _append_value(cmd, "phase1-target-long-fraction", phase1_cfg.get("target_long_fraction"))
    _append_bool_optional(cmd, "phase1-runtime-adaptive-enabled", phase1_cfg.get("runtime_adaptive_enabled"))
    _append_value(cmd, "phase1-runtime-aggressive-long-fraction", phase1_cfg.get("runtime_aggressive_long_fraction"))
    _append_value(cmd, "phase1-runtime-conservative-long-fraction", phase1_cfg.get("runtime_conservative_long_fraction"))
    _append_value(cmd, "phase1-runtime-aggressive-ingress-target-chunk", phase1_cfg.get("runtime_aggressive_ingress_target_chunk"))
    _append_value(cmd, "phase1-runtime-conservative-ingress-target-chunk", phase1_cfg.get("runtime_conservative_ingress_target_chunk"))
    _append_value(cmd, "phase1-runtime-queue-high-watermark", phase1_cfg.get("runtime_queue_high_watermark"))
    _append_value(cmd, "phase1-runtime-waiting-short-high-watermark", phase1_cfg.get("runtime_waiting_short_high_watermark"))
    _append_value(cmd, "phase1-runtime-wait-us-high-watermark", phase1_cfg.get("runtime_wait_us_high_watermark"))
    _append_value(cmd, "phase1-runtime-long-high-watermark", phase1_cfg.get("runtime_long_high_watermark"))
    _append_value(cmd, "phase1-runtime-urgency-discount", phase1_cfg.get("runtime_urgency_discount"))
    _append_value(cmd, "phase1-runtime-ema-alpha", phase1_cfg.get("runtime_ema_alpha"))

    _append_value(cmd, "phase12-phase2-gate-mode", phase12_cfg.get("phase2_gate_mode"))
    _append_value(cmd, "phase12-phase2-soft-ratio-scale", phase12_cfg.get("soft_ratio_scale"))
    _append_value(cmd, "phase12-phase2-soft-pressure-scale", phase12_cfg.get("soft_pressure_scale"))
    _append_value(cmd, "phase12-phase2-soft-min-long-prefill", phase12_cfg.get("soft_min_long_prefill"))
    _append_bool_optional(cmd, "phase12-phase2-soft-allow-mixed-decode", phase12_cfg.get("soft_allow_mixed_decode"))
    _append_value(cmd, "phase12-phase2-soft-recent-strength-floor", phase12_cfg.get("soft_recent_strength_floor"))
    _append_bool_optional(cmd, "phase12-phase2-soft-require-cashout-signal", phase12_cfg.get("soft_require_cashout_signal"))
    _append_value(cmd, "phase12-phase2-soft-recent-chunk-match-scale", phase12_cfg.get("soft_recent_chunk_match_scale"))
    _append_value(cmd, "phase12-phase2-soft-window-score-threshold", phase12_cfg.get("soft_window_score_threshold"))
    _append_value(cmd, "phase12-phase2-soft-window-recent-weight", phase12_cfg.get("soft_window_recent_weight"))
    _append_value(cmd, "phase12-phase2-soft-window-chunk-weight", phase12_cfg.get("soft_window_chunk_weight"))
    _append_value(cmd, "phase12-phase2-soft-window-pressure-weight", phase12_cfg.get("soft_window_pressure_weight"))
    _append_value(cmd, "phase12-phase2-soft-window-ratio-weight", phase12_cfg.get("soft_window_ratio_weight"))
    _append_value(cmd, "phase12-phase2-soft-window-decode-bonus", phase12_cfg.get("soft_window_decode_bonus"))

    _append_value(cmd, "phase2-dispatch-mode", phase2_cfg.get("dispatch_mode"))
    _append_bool_optional(cmd, "phase2-enable-mixed-prefill-decode", phase2_cfg.get("enable_mixed_prefill_decode"))
    _append_value(cmd, "phase2-min-hetero-ratio", phase2_cfg.get("min_hetero_ratio"))
    _append_value(cmd, "phase2-min-long-prefill", phase2_cfg.get("min_long_prefill"))
    _append_value(cmd, "phase2-min-pressure-ratio", phase2_cfg.get("min_pressure_ratio"))
    _append_bool_optional(
        cmd,
        "phase2-baseline-enable-chunked-prefill",
        phase2_cfg.get("baseline_enable_chunked_prefill"),
    )
    _append_bool_optional(cmd, "phase2-enable-scheduler-cashout", phase2_cfg.get("enable_scheduler_cashout"))
    _append_bool_optional(cmd, "phase2-enable-execution-escape", phase2_cfg.get("enable_execution_escape"))
    _append_value(cmd, "phase2-execution-escape-mode", phase2_cfg.get("execution_escape_mode"))
    _append_value(cmd, "phase2-execution-escape-spillover-cap", phase2_cfg.get("execution_escape_spillover_cap"))
    _append_value(cmd, "phase2-execution-escape-max-active", phase2_cfg.get("execution_escape_max_active"))
    _append_bool_optional(cmd, "phase2-runtime-adaptive-enabled", phase2_cfg.get("runtime_adaptive_enabled"))
    _append_value(cmd, "phase2-runtime-low-pressure-min-hetero-ratio", phase2_cfg.get("runtime_low_pressure_min_hetero_ratio"))
    _append_value(cmd, "phase2-runtime-high-pressure-min-hetero-ratio", phase2_cfg.get("runtime_high_pressure_min_hetero_ratio"))
    _append_value(cmd, "phase2-runtime-low-pressure-min-pressure-ratio", phase2_cfg.get("runtime_low_pressure_min_pressure_ratio"))
    _append_value(cmd, "phase2-runtime-high-pressure-min-pressure-ratio", phase2_cfg.get("runtime_high_pressure_min_pressure_ratio"))
    _append_value(cmd, "phase2-runtime-low-pressure-min-long-prefill", phase2_cfg.get("runtime_low_pressure_min_long_prefill"))
    _append_value(cmd, "phase2-runtime-high-pressure-min-long-prefill", phase2_cfg.get("runtime_high_pressure_min_long_prefill"))
    _append_value(cmd, "phase2-runtime-low-pressure-escape-spillover-cap", phase2_cfg.get("runtime_low_pressure_escape_spillover_cap"))
    _append_value(cmd, "phase2-runtime-high-pressure-escape-spillover-cap", phase2_cfg.get("runtime_high_pressure_escape_spillover_cap"))
    _append_value(cmd, "phase2-runtime-low-pressure-escape-max-active", phase2_cfg.get("runtime_low_pressure_escape_max_active"))
    _append_value(cmd, "phase2-runtime-high-pressure-escape-max-active", phase2_cfg.get("runtime_high_pressure_escape_max_active"))
    _append_value(cmd, "phase2-runtime-disable-execution-escape-below-pressure", phase2_cfg.get("runtime_disable_execution_escape_below_pressure"))

    _append_value(cmd, "out-json", result_json)

    vllm_mode = str(runtime_cfg.get("vllm_mode") or "v0").strip().lower()
    if vllm_mode not in {"v0", "v1"}:
        raise ValueError(f"Unsupported runtime.vllm_mode: {vllm_mode}")
    env = os.environ.copy()
    env["WAVESLICE_VLLM_MODE"] = vllm_mode
    env["VLLM_USE_V1"] = "1" if vllm_mode == "v1" else "0"
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    return cmd, env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a frozen single-case eval config through tests/evaluate_waveslice_claims.py.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    cmd, env = build_eval_invocation(
        config,
        out_json_override=args.out_json.strip() or None,
    )

    out_json = ""
    if "--out-json" in cmd:
        out_json = cmd[cmd.index("--out-json") + 1]
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"[FrozenEval] config={args.config}")
    if out_json:
        print(f"[FrozenEval] out_json={out_json}")
    print(f"[FrozenEval] env WAVESLICE_VLLM_MODE={env['WAVESLICE_VLLM_MODE']} VLLM_USE_V1={env['VLLM_USE_V1']}")
    print("[FrozenEval] command:")
    print("  " + shlex.join(cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, env=env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
