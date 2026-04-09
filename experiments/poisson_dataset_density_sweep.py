"""Sequential single-model dataset evaluation under Poisson arrivals.

This script builds thicker open-dataset workloads and evaluates one model
under several arrival-rate densities, one after another, to avoid GPU
cross-contamination.
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
from typing import Any

from config.experiment_catalog import get_model_specs
from config.experiment_catalog import safe_key as _safe_key
from experiments.model_assets import ensure_adapters as _ensure_adapters
from experiments.model_assets import resolve_local_snapshot as _resolve_local_snapshot


DEFAULT_DENSITIES = {
    "low": (2.0, 2.0),
    "mid": (6.0, 6.0),
    "high": (12.0, 12.0),
}


def _parse_densities(raw: str) -> list[tuple[str, float, float]]:
    if not raw.strip():
        return [(k, v1, v2) for k, (v1, v2) in DEFAULT_DENSITIES.items()]
    out: list[tuple[str, float, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, pair = item.split("=", 1)
        else:
            label, pair = item, item
        if ":" in pair:
            p1, p2 = pair.split(":", 1)
            out.append((label.strip(), float(p1), float(p2)))
        else:
            rate = float(pair)
            out.append((label.strip(), rate, rate))
    if not out:
        raise ValueError("no valid densities parsed")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential Poisson-arrival dataset sweep for one model.")
    parser.add_argument("--model", required=True, help="Model key from config/experiment_catalog.py")
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
    parser.add_argument("--densities", default="", help="Comma-separated densities like low=2:2,mid=6:6,high=12:12")
    parser.add_argument("--arrival-seed", type=int, default=7)
    parser.add_argument("--adapters-root", default=os.path.join("results", "synthetic_adapters"))
    parser.add_argument("--workload-root", default=os.path.join("results", "dataset_workloads_poisson"))
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    spec = get_model_specs(args.model)[0]
    densities = _parse_densities(args.densities)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or os.path.join("results", f"waveslice_poisson_density_{_safe_key(spec.key)}_{ts}")
    out_csv = args.out_csv or f"{results_dir}.csv"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(args.workload_root, exist_ok=True)

    local_snapshot = _resolve_local_snapshot(spec.model_id)
    model_path = spec.model_id if spec.trust_remote_code else (local_snapshot or spec.model_id)
    effective_max_model_len = spec.max_model_len_override or args.max_model_len

    adapter_dir = os.path.join(args.adapters_root, _safe_key(spec.key))
    adapter_a, adapter_b = _ensure_adapters(
        base_model_path=model_path,
        out_dir=adapter_dir,
        trust_remote_code=spec.trust_remote_code,
    )

    rows: list[dict[str, Any]] = []
    print(f"[PoissonSweep] model={spec.key} densities={densities}")
    for idx, (label, p1_rate, p2_rate) in enumerate(densities, start=1):
        print(f"\n[PoissonSweep] ({idx}/{len(densities)}) label={label} phase1_rate={p1_rate} phase2_rate={p2_rate}")
        try:
            out_prefix = os.path.join(args.workload_root, f"{_safe_key(spec.key)}_{_safe_key(label)}")
            workload_cmd = [
                sys.executable,
                "experiments/build_dataset_workload.py",
                "--model-path",
                model_path,
                "--out-prefix",
                out_prefix,
                "--max-prompt-tokens",
                str(max(16, effective_max_model_len - args.max_new_tokens - 16)),
                "--sample-count",
                str(args.sample_count),
                "--phase1-short-count",
                str(args.phase1_short_count),
                "--phase1-long-count",
                str(args.phase1_long_count),
                "--phase2-short-count",
                str(args.phase2_short_count),
                "--phase2-long-count",
                str(args.phase2_long_count),
                "--arrival-mode",
                "poisson",
                "--phase1-arrival-rate",
                str(p1_rate),
                "--phase2-arrival-rate",
                str(p2_rate),
                "--arrival-seed",
                str(args.arrival_seed + idx),
            ]
            if spec.trust_remote_code:
                workload_cmd.append("--trust-remote-code")
            env = os.environ.copy()
            env.setdefault("HF_ENDPOINT", "https://huggingface.co")
            workload_proc = subprocess.run(workload_cmd, capture_output=True, text=True, check=False, env=env)
            row: dict[str, Any] = {
                "density_label": label,
                "phase1_arrival_rate": p1_rate,
                "phase2_arrival_rate": p2_rate,
                "status": "failed" if workload_proc.returncode != 0 else "ok",
                "workload_stdout_tail": "\n".join((workload_proc.stdout or "").splitlines()[-20:]),
                "workload_stderr_tail": "\n".join((workload_proc.stderr or "").splitlines()[-20:]),
            }
            if workload_proc.returncode != 0:
                row["error"] = f"build_dataset_workload exited with code {workload_proc.returncode}"
                rows.append(row)
                continue

            req_json = f"{out_prefix}_requests.json"
            lora_req_json = f"{out_prefix}_lora_requests.json"
            meta_json = f"{out_prefix}_meta.json"
            out_json = os.path.join(results_dir, f"{_safe_key(spec.key)}_{_safe_key(label)}_eval.json")
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
                str(args.max_new_tokens),
                "--timeout-sec",
                str(args.timeout_sec),
                "--warmup-iters",
                str(args.warmup_iters),
                "--repeats",
                str(args.repeats),
                "--max-model-len",
                str(effective_max_model_len),
                "--max-num-batched-tokens",
                str(args.max_num_batched_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--phase2-dispatch-mode",
                args.phase2_dispatch_mode,
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
            if eval_proc.returncode == 0:
                summary = json.load(open(out_json, "r", encoding="utf-8"))
                meta = json.load(open(meta_json, "r", encoding="utf-8"))
                phase12 = summary.get("phase12", {})
                row["phase12_ttft_improve_mean"] = (phase12.get("ttft_improve_ratio") or {}).get("mean")
                row["phase12_wall_improve_mean"] = (phase12.get("round_wall_improve_ratio") or {}).get("mean")
                row["phase12_slowdown_improve_mean"] = (phase12.get("slowdown_improve_ratio") or {}).get("mean")
                row["phase12_apply_mean"] = (phase12.get("phase2_apply_ratio") or {}).get("mean")
                row["phase12_incremental_error_mean"] = (phase12.get("incremental_error_rate") or {}).get("mean")
                row["phase1_request_count"] = meta.get("phase1_request_count")
                row["phase2_request_count"] = meta.get("phase2_request_count")
                row["phase1_last_arrival_s"] = meta.get("phase1_last_arrival_s")
                row["phase2_last_arrival_s"] = meta.get("phase2_last_arrival_s")
            else:
                row["error"] = f"evaluate_waveslice_claims exited with code {eval_proc.returncode}"
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "density_label": label,
                    "phase1_arrival_rate": p1_rate,
                    "phase2_arrival_rate": p2_rate,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[PoissonSweep] CSV written: {out_csv}")
    print(f"[PoissonSweep] JSON dir: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
