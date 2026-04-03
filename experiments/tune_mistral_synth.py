"""Sequential Mistral synthetic-load tuner for Wave-Slice.

This script is intended for paper-facing experiment tuning before we scale to
the full 8-model suite. It runs a small set of curated Mistral workloads
sequentially and aggregates the key metrics into one JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from config.experiment_catalog import DEFAULT_SYNTHETIC_ADAPTER_PRESETS

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")


DEFAULT_MODEL_PATH = "mistralai/Mistral-7B-v0.1"
DEFAULT_ADAPTERS_ROOT = os.path.join("results", "synthetic_adapters")
DEFAULT_ADAPTER_A = os.path.join(
    DEFAULT_ADAPTERS_ROOT,
    "mistral-7b-v0.1",
    f"adapter_rank{DEFAULT_SYNTHETIC_ADAPTER_PRESETS[0].rank}_seed{DEFAULT_SYNTHETIC_ADAPTER_PRESETS[0].seed}",
)
DEFAULT_ADAPTER_B = os.path.join(
    DEFAULT_ADAPTERS_ROOT,
    "mistral-7b-v0.1",
    f"adapter_rank{DEFAULT_SYNTHETIC_ADAPTER_PRESETS[1].rank}_seed{DEFAULT_SYNTHETIC_ADAPTER_PRESETS[1].seed}",
)


@dataclass(frozen=True)
class SweepCase:
    name: str
    short_a_repeat: int
    short_b_repeat: int
    long_repeat: int
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    max_new_tokens: int = 24
    warmup_iters: int = 2
    repeats: int = 3
    timeout_sec: int = 300
    phase1_objective_mode: str = "fair_escape"
    include_strict: bool = False


DEFAULT_CASES: list[SweepCase] = [
    SweepCase(
        name="baseline_gap_2304",
        short_a_repeat=10,
        short_b_repeat=10,
        long_repeat=90,
        max_model_len=2304,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.88,
        max_new_tokens=32,
    ),
    SweepCase(
        name="gap_3072_fair_escape",
        short_a_repeat=4,
        short_b_repeat=8,
        long_repeat=140,
        max_model_len=3072,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.88,
        phase1_objective_mode="fair_escape",
    ),
    SweepCase(
        name="gap_3072_pure_gain",
        short_a_repeat=4,
        short_b_repeat=8,
        long_repeat=140,
        max_model_len=3072,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.88,
        phase1_objective_mode="pure_gain",
    ),
    SweepCase(
        name="extreme_3584_strict",
        short_a_repeat=2,
        short_b_repeat=6,
        long_repeat=180,
        max_model_len=3584,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.88,
        include_strict=True,
    ),
]


def _build_cmd(
    *,
    case: SweepCase,
    model_path: str,
    adapter_a: str,
    adapter_b: str,
    out_json: str,
    warmup_override: int | None = None,
    repeats_override: int | None = None,
    timeout_override: int | None = None,
) -> list[str]:
    warmup_iters = case.warmup_iters if warmup_override is None else warmup_override
    repeats = case.repeats if repeats_override is None else repeats_override
    timeout_sec = case.timeout_sec if timeout_override is None else timeout_override
    cmd = [
        sys.executable,
        "tests/evaluate_waveslice_claims.py",
        "--model-name",
        "Mistral-7B-v0.1",
        "--model-path",
        model_path,
        "--adapter-a",
        adapter_a,
        "--adapter-b",
        adapter_b,
        "--max-new-tokens",
        str(case.max_new_tokens),
        "--warmup-iters",
        str(warmup_iters),
        "--repeats",
        str(repeats),
        "--short-a-repeat",
        str(case.short_a_repeat),
        "--short-b-repeat",
        str(case.short_b_repeat),
        "--long-repeat",
        str(case.long_repeat),
        "--max-model-len",
        str(case.max_model_len),
        "--max-num-batched-tokens",
        str(case.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(case.gpu_memory_utilization),
        "--phase1-objective-mode",
        case.phase1_objective_mode,
        "--timeout-sec",
        str(timeout_sec),
        "--out-json",
        out_json,
    ]
    if case.include_strict:
        cmd.append("--include-strict")
    return cmd


def _extract_metric(block: dict[str, Any], key: str) -> Any:
    return (block.get(key) or {}).get("mean")


def _case_summary(payload: dict[str, Any]) -> dict[str, Any]:
    phase1 = payload.get("phase1", {})
    phase2 = payload.get("phase2", {})
    strict = payload.get("phase2_strict", {})
    return {
        "token_lengths": payload.get("token_lengths"),
        "phase1_ttft_improve_mean": _extract_metric(phase1, "ttft_improve_ratio"),
        "phase1_wall_improve_mean": _extract_metric(phase1, "round_wall_improve_ratio"),
        "phase1_error_mean": _extract_metric(phase1, "error_rate"),
        "phase1_noise_mean": _extract_metric(phase1, "baseline_noise_error_rate"),
        "phase1_incremental_error_mean": _extract_metric(phase1, "incremental_error_rate"),
        "phase1_apply_mean": _extract_metric(phase1, "scheduler_apply_ratio"),
        "phase2_ttft_improve_mean": _extract_metric(phase2, "ttft_improve_ratio"),
        "phase2_slowdown_improve_mean": _extract_metric(phase2, "slowdown_improve_ratio"),
        "phase2_wall_improve_mean": _extract_metric(phase2, "round_wall_improve_ratio"),
        "phase2_error_mean": _extract_metric(phase2, "wave_error_rate"),
        "phase2_noise_mean": _extract_metric(phase2, "baseline_noise_error_rate"),
        "phase2_incremental_error_mean": _extract_metric(phase2, "incremental_error_rate"),
        "phase2_apply_mean": _extract_metric(phase2, "phase2_apply_ratio"),
        "phase2_strict_ttft_improve_mean": _extract_metric(strict, "ttft_improve_ratio"),
        "phase2_strict_slowdown_improve_mean": _extract_metric(strict, "slowdown_improve_ratio"),
        "phase2_strict_wall_improve_mean": _extract_metric(strict, "round_wall_improve_ratio"),
        "phase2_strict_error_mean": _extract_metric(strict, "error_rate"),
        "phase2_strict_incremental_error_mean": _extract_metric(strict, "incremental_error_rate"),
        "phase2_strict_apply_mean": _extract_metric(strict, "apply_ratio"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential Mistral synthetic tuning sweep.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-a", default=DEFAULT_ADAPTER_A)
    parser.add_argument("--adapter-b", default=DEFAULT_ADAPTER_B)
    parser.add_argument(
        "--adapters-root",
        default=DEFAULT_ADAPTERS_ROOT,
        help="Directory for synthetic adapters when using the default adapter paths.",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case names. Defaults to all curated cases.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Default: results/mistral_tuning_<ts>.json",
    )
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=None)
    args = parser.parse_args()

    if args.adapter_a == DEFAULT_ADAPTER_A:
        args.adapter_a = os.path.join(
            args.adapters_root,
            "mistral-7b-v0.1",
            f"adapter_rank{DEFAULT_SYNTHETIC_ADAPTER_PRESETS[0].rank}_seed{DEFAULT_SYNTHETIC_ADAPTER_PRESETS[0].seed}",
        )
    if args.adapter_b == DEFAULT_ADAPTER_B:
        args.adapter_b = os.path.join(
            args.adapters_root,
            "mistral-7b-v0.1",
            f"adapter_rank{DEFAULT_SYNTHETIC_ADAPTER_PRESETS[1].rank}_seed{DEFAULT_SYNTHETIC_ADAPTER_PRESETS[1].seed}",
        )

    case_map = {case.name: case for case in DEFAULT_CASES}
    if args.cases:
        wanted = [name.strip() for name in args.cases.split(",") if name.strip()]
        cases = [case_map[name] for name in wanted]
    else:
        cases = DEFAULT_CASES

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = args.out_json or f"results/mistral_tuning_{ts}.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    case_dir = os.path.join("results", "tuning_cases")
    os.makedirs(case_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    print(f"[Tune] cases={len(cases)}")
    for idx, case in enumerate(cases, start=1):
        case_json = os.path.join(case_dir, f"{case.name}_{ts}.json")
        cmd = _build_cmd(
            case=case,
            model_path=args.model_path,
            adapter_a=args.adapter_a,
            adapter_b=args.adapter_b,
            out_json=case_json,
            warmup_override=args.warmup_iters,
            repeats_override=args.repeats,
            timeout_override=args.timeout_sec,
        )
        print(f"[Tune] ({idx}/{len(cases)}) {case.name}")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        row: dict[str, Any] = {
            "case": case.name,
            "returncode": proc.returncode,
            "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-20:]),
            "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-20:]),
            "result_json": case_json,
        }
        if proc.returncode == 0 and os.path.exists(case_json):
            with open(case_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            row.update(_case_summary(payload))
        rows.append(row)

    payload = {
        "model_path": args.model_path,
        "adapter_a": args.adapter_a,
        "adapter_b": args.adapter_b,
        "cases": [case.name for case in cases],
        "results": rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Tune] output={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
