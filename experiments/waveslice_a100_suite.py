"""Wave-Slice multi-model experiment suite for A100-class GPUs.

Focus:
- Same-base + heterogeneous LoRA concurrent serving.
- Baseline vs Wave-Slice comparison.
- Non-LLaMA models only by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

from config.experiment_catalog import (
    DEFAULT_EXPERIMENT_MODELS,
    DEFAULT_SYNTHETIC_ADAPTER_PRESETS,
    ExperimentModelSpec as ModelSpec,
    get_model_specs,
    safe_key,
)
from engine.runtime_bootstrap import bootstrap_vllm_runtime
from tools.synthetic_lora_builder import AdapterSpec, build_synthetic_adapters

bootstrap_vllm_runtime()


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _resolve_local_snapshot(model_id: str) -> Optional[str]:
    home = Path.home()
    hub_dir = home / ".cache" / "huggingface" / "hub"
    repo_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = hub_dir / repo_name / "snapshots"
    if not snapshots_dir.exists():
        return None
    dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    # pick the most recent *usable* snapshot (must include config.json).
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in dirs:
        if (snap / "config.json").exists():
            return str(snap)
    return None


def _ensure_adapters(
    *,
    base_model_path: str,
    out_dir: str,
    trust_remote_code: bool,
) -> tuple[str, str]:
    preset_a, preset_b = DEFAULT_SYNTHETIC_ADAPTER_PRESETS[:2]
    path_a = os.path.join(out_dir, preset_a.name)
    path_b = os.path.join(out_dir, preset_b.name)
    marker_a = os.path.join(path_a, "adapter_config.json")
    marker_b = os.path.join(path_b, "adapter_config.json")
    if os.path.exists(marker_a) and os.path.exists(marker_b):
        return path_a, path_b

    generated = build_synthetic_adapters(
        base_model=base_model_path,
        out_dir=out_dir,
        specs=[
            AdapterSpec(
                name=spec.name,
                rank=spec.rank,
                alpha=spec.alpha,
                seed=spec.seed,
                init_std=spec.init_std,
            )
            for spec in DEFAULT_SYNTHETIC_ADAPTER_PRESETS
        ],
        trust_remote_code=trust_remote_code,
    )
    return generated[0], generated[1]


def _run_model(
    *,
    spec: ModelSpec,
    adapters_root: str,
    timeout_sec: int,
    max_new_tokens: int,
    warmup_iters: int,
    repeats: int,
    short_repeat: int,
    long_repeat: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    include_strict: bool,
    include_phase12: bool,
    phase2_dispatch_mode: str,
    results_dir: str,
    dry_run: bool,
) -> dict[str, Any]:
    local_snapshot = _resolve_local_snapshot(spec.model_id)
    model_path = spec.model_id if spec.trust_remote_code else (local_snapshot or spec.model_id)
    adapter_dir = os.path.join(adapters_root, _safe_key(spec.key))

    if dry_run:
        return {
            "model": spec.model_id,
            "lut_name": spec.lut_name,
            "model_path": model_path,
            "adapter_dir": adapter_dir,
            "status": "dry_run",
        }

    adapter_a, adapter_b = _ensure_adapters(
        base_model_path=model_path,
        out_dir=adapter_dir,
        trust_remote_code=spec.trust_remote_code,
    )

    effective_max_model_len = spec.max_model_len_override or max_model_len

    row: dict[str, Any] = {
        "model": spec.model_id,
        "lut_name": spec.lut_name,
        "model_path": model_path,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "effective_max_model_len": effective_max_model_len,
    }
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, f"{_safe_key(spec.key)}_repeated_eval.json")
    cmd = [
        sys.executable,
        "tests/evaluate_waveslice_claims.py",
        "--model-name",
        spec.lut_name,
        "--model-path",
        model_path,
        "--adapter-a",
        adapter_a,
        "--adapter-b",
        adapter_b,
        "--max-new-tokens",
        str(max_new_tokens),
        "--timeout-sec",
        str(timeout_sec),
        "--warmup-iters",
        str(warmup_iters),
        "--repeats",
        str(repeats),
        "--short-repeat",
        str(short_repeat),
        "--long-repeat",
        str(long_repeat),
        "--max-model-len",
        str(effective_max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--phase2-dispatch-mode",
        phase2_dispatch_mode,
        "--out-json",
        out_json,
    ]
    if include_strict:
        cmd.append("--include-strict")
    if include_phase12:
        cmd.append("--include-phase12")
    if spec.trust_remote_code:
        cmd.append("--trust-remote-code")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    row["status"] = "ok" if proc.returncode == 0 else "failed"
    row["stdout_tail"] = "\n".join((proc.stdout or "").splitlines()[-20:])
    row["stderr_tail"] = "\n".join((proc.stderr or "").splitlines()[-20:])
    row["result_json"] = out_json

    if proc.returncode != 0:
        row["error"] = f"evaluate_waveslice_claims exited with code {proc.returncode}"
        return row

    with open(out_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    row["token_lengths"] = summary.get("token_lengths")
    phase1 = summary.get("phase1", {})
    phase2 = summary.get("phase2", {})
    strict = summary.get("phase2_strict", {})
    phase12 = summary.get("phase12", {})
    row["phase1_ttft_improve_mean"] = (phase1.get("ttft_improve_ratio") or {}).get("mean")
    row["phase1_wall_improve_mean"] = (phase1.get("round_wall_improve_ratio") or {}).get("mean")
    row["phase1_error_mean"] = (phase1.get("error_rate") or {}).get("mean")
    row["phase1_apply_mean"] = (phase1.get("scheduler_apply_ratio") or {}).get("mean")
    row["phase2_ttft_improve_mean"] = (phase2.get("ttft_improve_ratio") or {}).get("mean")
    row["phase2_slowdown_improve_mean"] = (phase2.get("slowdown_improve_ratio") or {}).get("mean")
    row["phase2_wall_improve_mean"] = (phase2.get("round_wall_improve_ratio") or {}).get("mean")
    row["phase2_error_mean"] = (phase2.get("wave_error_rate") or {}).get("mean")
    row["phase2_noise_mean"] = (phase2.get("baseline_noise_error_rate") or {}).get("mean")
    row["phase2_incremental_error_mean"] = (phase2.get("incremental_error_rate") or {}).get("mean")
    row["phase2_apply_mean"] = (phase2.get("phase2_apply_ratio") or {}).get("mean")
    if include_phase12:
        row["phase12_ttft_improve_mean"] = (phase12.get("ttft_improve_ratio") or {}).get("mean")
        row["phase12_slowdown_improve_mean"] = (phase12.get("slowdown_improve_ratio") or {}).get("mean")
        row["phase12_wall_improve_mean"] = (phase12.get("round_wall_improve_ratio") or {}).get("mean")
        row["phase12_error_mean"] = (phase12.get("wave_error_rate") or {}).get("mean")
        row["phase12_noise_mean"] = (phase12.get("baseline_noise_error_rate") or {}).get("mean")
        row["phase12_incremental_error_mean"] = (phase12.get("incremental_error_rate") or {}).get("mean")
        row["phase12_apply_mean"] = (phase12.get("phase2_apply_ratio") or {}).get("mean")
    if include_strict:
        row["phase2_strict_ttft_improve_mean"] = (strict.get("ttft_improve_ratio") or {}).get("mean")
        row["phase2_strict_slowdown_improve_mean"] = (strict.get("slowdown_improve_ratio") or {}).get("mean")
        row["phase2_strict_wall_improve_mean"] = (strict.get("round_wall_improve_ratio") or {}).get("mean")
        row["phase2_strict_error_mean"] = (strict.get("error_rate") or {}).get("mean")
        row["phase2_strict_incremental_error_mean"] = (strict.get("incremental_error_rate") or {}).get("mean")
        row["phase2_strict_apply_mean"] = (strict.get("apply_ratio") or {}).get("mean")
    return row


def _write_csv(rows: list[dict[str, Any]], out_csv: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Wave-Slice A100 suite on multiple non-LLaMA models.")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys. Defaults to all 8 models.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Per-round timeout in seconds.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Per-request decode length. Use >=32 for production-like experiments.",
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--short-repeat", type=int, default=24)
    parser.add_argument("--long-repeat", type=int, default=360)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--include-strict", action="store_true")
    parser.add_argument("--include-phase12", action="store_true")
    parser.add_argument(
        "--phase2-dispatch-mode",
        choices=["synchronized", "async_experimental"],
        default="synchronized",
    )
    parser.add_argument(
        "--adapters-root",
        default=os.path.join("results", "synthetic_adapters"),
        help="Directory to store synthetic adapters.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to store per-model JSON results. Default: results/waveslice_a100_suite_<ts>",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Default: results/waveslice_a100_suite_<ts>.csv",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan without running models.")
    parser.add_argument("--list-models", action="store_true", help="List available model keys and exit.")
    args = parser.parse_args()

    if args.list_models:
        print("Available model keys:")
        for spec in DEFAULT_EXPERIMENT_MODELS:
            print(f"  {spec.key:24s} -> {spec.model_id}")
        return 0

    models = get_model_specs(args.models)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = args.out_csv or f"results/waveslice_a100_suite_{ts}.csv"
    results_dir = args.results_dir or f"results/waveslice_a100_suite_{ts}"

    rows: list[dict[str, Any]] = []
    print(
        "[Suite] "
        f"models={len(models)} dry_run={args.dry_run} "
        f"warmup={args.warmup_iters} repeats={args.repeats} "
        f"decode={args.max_new_tokens}"
    )
    for idx, spec in enumerate(models, start=1):
        print(f"\n[Suite] ({idx}/{len(models)}) {spec.key} | {spec.model_id}")
        try:
            row = _run_model(
                spec=spec,
                adapters_root=args.adapters_root,
                timeout_sec=args.timeout_sec,
                max_new_tokens=args.max_new_tokens,
                warmup_iters=args.warmup_iters,
                repeats=args.repeats,
                short_repeat=args.short_repeat,
                long_repeat=args.long_repeat,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                include_strict=args.include_strict,
                include_phase12=args.include_phase12,
                phase2_dispatch_mode=args.phase2_dispatch_mode,
                results_dir=results_dir,
                dry_run=args.dry_run,
            )
            rows.append(row)
            print(
                "[Suite] done | "
                f"phase1_ttft={row.get('phase1_ttft_improve_mean')} "
                f"phase2_ttft={row.get('phase2_ttft_improve_mean')} "
                f"phase2_slow={row.get('phase2_slowdown_improve_mean')} "
                f"phase12_ttft={row.get('phase12_ttft_improve_mean')} "
                f"phase12_wall={row.get('phase12_wall_improve_mean')} "
                f"status={row.get('status')}"
            )
        except Exception as exc:
            print(f"[Suite] FAIL: {exc}")
            print(traceback.format_exc())
            rows.append(
                {
                    "model": spec.model_id,
                    "lut_name": spec.lut_name,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    _write_csv(rows, out_csv)
    print(f"\n[Suite] CSV written: {out_csv}")
    print(f"[Suite] JSON dir: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
