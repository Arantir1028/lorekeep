"""Wave-Slice multi-model dataset-driven experiment suite.

This suite reuses the repeated evaluation harness, but replaces the synthetic
prompt templates with mixed open-dataset workloads built from:
- LongBench
- UltraChat200k
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from config.experiment_catalog import DEFAULT_DATASET_SUITE_KEYS, get_model_specs
from experiments.waveslice_a100_suite import (
    ModelSpec,
    _ensure_adapters,
    _resolve_local_snapshot,
    safe_key as _safe_key,
)


def _select_models(keys: str) -> list[ModelSpec]:
    selected = get_model_specs(keys)
    if not selected:
        raise ValueError(f"no models selected from keys={keys!r}")
    return selected


def _run_model(
    *,
    spec: ModelSpec,
    adapters_root: str,
    workload_root: str,
    results_dir: str,
    timeout_sec: int,
    max_new_tokens: int,
    warmup_iters: int,
    repeats: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    phase2_dispatch_mode: str,
    sample_count: int,
    phase1_short_count: int,
    phase1_long_count: int,
    phase2_short_count: int,
    phase2_long_count: int,
) -> dict[str, Any]:
    local_snapshot = _resolve_local_snapshot(spec.model_id)
    model_path = spec.model_id if spec.trust_remote_code else (local_snapshot or spec.model_id)
    effective_max_model_len = spec.max_model_len_override or max_model_len

    adapter_dir = os.path.join(adapters_root, _safe_key(spec.key))
    adapter_a, adapter_b = _ensure_adapters(
        base_model_path=model_path,
        out_dir=adapter_dir,
        trust_remote_code=spec.trust_remote_code,
    )

    os.makedirs(workload_root, exist_ok=True)
    out_prefix = os.path.join(workload_root, _safe_key(spec.key))
    workload_cmd = [
        sys.executable,
        "experiments/build_dataset_workload.py",
        "--model-path",
        model_path,
        "--out-prefix",
        out_prefix,
        "--max-prompt-tokens",
        str(max(16, effective_max_model_len - max_new_tokens - 16)),
        "--sample-count",
        str(sample_count),
        "--phase1-short-count",
        str(phase1_short_count),
        "--phase1-long-count",
        str(phase1_long_count),
        "--phase2-short-count",
        str(phase2_short_count),
        "--phase2-long-count",
        str(phase2_long_count),
    ]
    if spec.trust_remote_code:
        workload_cmd.append("--trust-remote-code")
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://huggingface.co")
    workload_proc = subprocess.run(workload_cmd, capture_output=True, text=True, check=False, env=env)

    row: dict[str, Any] = {
        "model": spec.model_id,
        "lut_name": spec.lut_name,
        "model_path": model_path,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "effective_max_model_len": effective_max_model_len,
        "workload_stdout_tail": "\n".join((workload_proc.stdout or "").splitlines()[-20:]),
        "workload_stderr_tail": "\n".join((workload_proc.stderr or "").splitlines()[-20:]),
    }
    if workload_proc.returncode != 0:
        row["status"] = "failed"
        row["error"] = f"build_dataset_workload exited with code {workload_proc.returncode}"
        return row

    req_json = f"{out_prefix}_requests.json"
    lora_req_json = f"{out_prefix}_lora_requests.json"
    meta_json = f"{out_prefix}_meta.json"
    out_json = os.path.join(results_dir, f"{_safe_key(spec.key)}_dataset_eval.json")

    eval_cmd = [
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
        "--max-model-len",
        str(effective_max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--phase2-dispatch-mode",
        phase2_dispatch_mode,
        "--include-phase12",
        "--requests-json",
        req_json,
        "--lora-requests-json",
        lora_req_json,
        "--out-json",
        out_json,
    ]
    if spec.trust_remote_code:
        eval_cmd.append("--trust-remote-code")
    eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True, check=False)
    row["status"] = "ok" if eval_proc.returncode == 0 else "failed"
    row["stdout_tail"] = "\n".join((eval_proc.stdout or "").splitlines()[-20:])
    row["stderr_tail"] = "\n".join((eval_proc.stderr or "").splitlines()[-20:])
    row["result_json"] = out_json
    row["workload_meta_json"] = meta_json

    if eval_proc.returncode != 0:
        row["error"] = f"evaluate_waveslice_claims exited with code {eval_proc.returncode}"
        return row

    summary = json.load(open(out_json, "r", encoding="utf-8"))
    meta = json.load(open(meta_json, "r", encoding="utf-8"))
    phase12 = summary.get("phase12", {})
    row["token_lengths"] = summary.get("request_token_lengths")
    row["dataset_short_a_tokens"] = meta.get("short_a_tokens")
    row["dataset_short_b_tokens"] = meta.get("short_b_tokens")
    row["dataset_long_a_tokens"] = meta.get("long_a_tokens")
    row["dataset_long_b_tokens"] = meta.get("long_b_tokens")
    row["phase1_request_count"] = meta.get("phase1_request_count")
    row["phase2_request_count"] = meta.get("phase2_request_count")
    row["phase12_ttft_improve_mean"] = (phase12.get("ttft_improve_ratio") or {}).get("mean")
    row["phase12_slowdown_improve_mean"] = (phase12.get("slowdown_improve_ratio") or {}).get("mean")
    row["phase12_wall_improve_mean"] = (phase12.get("round_wall_improve_ratio") or {}).get("mean")
    row["phase12_incremental_error_mean"] = (phase12.get("incremental_error_rate") or {}).get("mean")
    row["phase12_apply_mean"] = (phase12.get("phase2_apply_ratio") or {}).get("mean")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset-driven Wave-Slice suite.")
    parser.add_argument("--models", default=",".join(DEFAULT_DATASET_SUITE_KEYS))
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--phase2-dispatch-mode", default="synchronized")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--phase1-short-count", type=int, default=24)
    parser.add_argument("--phase1-long-count", type=int, default=8)
    parser.add_argument("--phase2-short-count", type=int, default=24)
    parser.add_argument("--phase2-long-count", type=int, default=12)
    parser.add_argument("--adapters-root", default=os.path.join("results", "synthetic_adapters"))
    parser.add_argument("--workload-root", default=os.path.join("results", "dataset_workloads"))
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    models = _select_models(args.models)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or os.path.join("results", f"waveslice_dataset_suite_{ts}")
    out_csv = args.out_csv or f"{results_dir}.csv"
    os.makedirs(results_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    print(
        f"[DatasetSuite] models={len(models)} warmup={args.warmup_iters} repeats={args.repeats} "
        f"decode={args.max_new_tokens} sample_count={args.sample_count}"
    )
    for idx, spec in enumerate(models, start=1):
        print(f"\n[DatasetSuite] ({idx}/{len(models)}) {spec.key} | {spec.model_id}")
        try:
            row = _run_model(
                spec=spec,
                adapters_root=args.adapters_root,
                workload_root=args.workload_root,
                results_dir=results_dir,
                timeout_sec=args.timeout_sec,
                max_new_tokens=args.max_new_tokens,
                warmup_iters=args.warmup_iters,
                repeats=args.repeats,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                phase2_dispatch_mode=args.phase2_dispatch_mode,
                sample_count=args.sample_count,
                phase1_short_count=args.phase1_short_count,
                phase1_long_count=args.phase1_long_count,
                phase2_short_count=args.phase2_short_count,
                phase2_long_count=args.phase2_long_count,
            )
        except Exception as exc:
            row = {
                "model": spec.model_id,
                "lut_name": spec.lut_name,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        print(
            "[DatasetSuite] done | "
            f"phase12_ttft={row.get('phase12_ttft_improve_mean')} "
            f"phase12_wall={row.get('phase12_wall_improve_mean')} "
            f"phase12_slow={row.get('phase12_slowdown_improve_mean')} "
            f"status={row.get('status')}"
        )
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[DatasetSuite] CSV written: {out_csv}")
    print(f"[DatasetSuite] JSON dir: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
