"""Rebuild the paper's fixed-chunking observations without loading CUCUMIS hooks."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.model_assets import ensure_model_available
from experiments.openworkload_models import resolve_model_entry
from experiments.openworkload_support import apply_hf_resource_env, resource_policy
from experiments.result_io import read_json as _load_json, resolve, write_json as _write_json
from experiments.run_frozen_eval_config import build_eval_invocation


def _phase1_prompt(short: bool, repeat: int, index: int = 0) -> str:
    text = (
        "Interactive assistant turn. Keep the answer concise, direct, and faithful. A serving system may receive many such requests while longer document jobs are also active. "
        if short
        else "Long-context serving workloads mix summarization, long-document QA, and instruction following. When these requests share one GPU with short interactive turns, the scheduler must decide which request keeps making progress. "
    ) * max(1, repeat)
    task = (
        f"Task {index}: rewrite the sentence in one natural English sentence with the same meaning. Sentence: 'Low latency matters most when the user is waiting for the next turn.'"
        if short
        else "Task: write exactly one sentence summarizing the passage without bullet points or headings."
    )
    return text + task


def _make_phase1_requests(
    *,
    pattern: str,
    short_count: int,
    short_prompt_repeat: int,
    long_prompt_repeat: int,
    short_start_s: float,
    short_gap_s: float,
    sequential_delay_s: float,
    long_count: int = 1,
    long_gap_s: float = 0.0,
) -> list[dict[str, Any]]:
    if long_count < 1 or pattern not in {"sequential", "long_first", "short_first", "interleaved"}:
        raise ValueError(f"invalid Phase-I workload: pattern={pattern}, long_count={long_count}")
    long_start = short_start_s + short_count * short_gap_s if pattern == "short_first" else 0.0
    short_start = (
        sequential_delay_s
        if pattern == "sequential"
        else 0.0
        if pattern == "short_first"
        else short_start_s
    )
    requests = [
        {
            "req_id": f"long_{index:02d}",
            "prompt": _phase1_prompt(False, long_prompt_repeat),
            "is_short": False,
            "arrival_offset_s": long_start + index * long_gap_s,
        }
        for index in range(long_count)
    ] + [
        {
            "req_id": f"short_{index:02d}",
            "prompt": _phase1_prompt(True, short_prompt_repeat, index),
            "is_short": True,
            "arrival_offset_s": short_start + index * short_gap_s,
        }
        for index in range(short_count)
    ]
    return sorted(requests, key=lambda request: request["arrival_offset_s"])


def _p99(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.99 * len(ordered)) - 1)]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _resolve_model(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(config.get("model") or {})
    resolved = resolve_model_entry(model_cfg)
    resources = resource_policy(config)
    snapshot = ensure_model_available(
        resolved.model_id,
        auto_download=bool(resources["auto_download"]),
        local_files_only=bool(resources["offline"]),
    )
    if resolved.model_path_mode == "local_snapshot_required" and (not snapshot):
        raise FileNotFoundError(
            f"The fixed observation configuration requires a local snapshot of {resolved.model_id}."
        )
    model_path = snapshot or resolved.model_id
    return {
        "key": resolved.key,
        "model_id": resolved.model_id,
        "model_name": resolved.lut_name,
        "model_path": model_path,
        "trust_remote_code": resolved.trust_remote_code,
        "max_model_len": int(
            model_cfg.get("max_model_len") or resolved.max_model_len_override or 3072
        ),
    }


def _build_trace(workload: dict[str, Any], long_count: int) -> list[dict[str, Any]]:
    return _make_phase1_requests(
        pattern=str(workload.get("pattern", "long_first")),
        short_count=int(workload.get("short_count", 6)),
        short_prompt_repeat=int(workload.get("short_prompt_repeat", 2)),
        long_prompt_repeat=int(workload.get("long_prompt_repeat", 70)),
        short_start_s=float(workload.get("short_start_s", 0.1)),
        short_gap_s=float(workload.get("short_gap_s", 0.04)),
        sequential_delay_s=float(workload.get("sequential_delay_s", 6.0)),
        long_count=long_count,
        long_gap_s=float(workload.get("long_gap_s", 0.0)),
    )


def _summarize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    rows = list(raw.get("chunked_rows") or [])
    timings = [
        timing
        for row in rows
        for timing in (row.get("request_timings") or {}).values()
        if isinstance(timing, dict)
    ]

    def numbers(items: list[dict[str, Any]], key: str) -> list[float]:
        return [
            float(value) for item in items if isinstance((value := item.get(key)), (int, float))
        ]

    short = [timing for timing in timings if timing.get("is_short")]
    scheduled_short_ttft = numbers(short, "scheduled_first_latency_ms")
    observed_short_ttft = numbers(short, "first_latency_ms")
    all_scheduled_ttft = numbers(timings, "scheduled_first_latency_ms")
    round_wall_ms, completed, total = (
        numbers(rows, key) for key in ("round_wall_ms", "finished_requests", "total_requests")
    )
    return {
        "repeat_count": len(rows),
        "short_scheduled_ttft_p99_ms": _p99(scheduled_short_ttft),
        "short_observed_ttft_p99_ms": _p99(observed_short_ttft),
        "all_scheduled_ttft_p99_ms": _p99(all_scheduled_ttft),
        "round_wall_ms_mean": _mean(round_wall_ms),
        "round_wall_ms_p99": _p99(round_wall_ms),
        "finished_requests_mean": _mean(completed),
        "total_requests_mean": _mean(total),
        "timed_out": any(row.get("timed_out") for row in rows),
    }


def _run_fixed_baseline(
    *,
    run_root: Path,
    case_name: str,
    model: dict[str, Any],
    evaluation: dict[str, Any],
    token_budget: int,
    requests: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    workloads = run_root / "workloads"
    raw_dir = run_root / "raw"
    logs = run_root / "logs"
    request_path = workloads / f"{case_name}_requests.json"
    raw_path = raw_dir / f"{case_name}.json"
    stdout_path = logs / f"{case_name}.stdout.log"
    stderr_path = logs / f"{case_name}.stderr.log"
    _write_json(request_path, requests)
    case_config = {
        "evaluator": "tests/evaluate_waveslice_claims.py",
        "baseline_only": True,
        "skip_phase2": True,
        "include_phase12": False,
        "model": {"name": model["model_name"], "path": model["model_path"]},
        "workload": {"requests_json": str(request_path)},
        "runtime": {
            "python_bin": sys.executable,
            "warmup_iters": int(evaluation.get("warmup_iters", 1)),
            "repeats": int(evaluation.get("repeats", 5)),
            "timeout_sec": int(evaluation.get("timeout_sec", 360)),
            "max_new_tokens": int(evaluation.get("max_new_tokens", 64)),
            "max_model_len": int(model["max_model_len"]),
            "max_num_batched_tokens": token_budget,
            "max_num_partial_prefills": int(evaluation.get("max_num_partial_prefills", 1)),
            "max_long_partial_prefills": int(evaluation.get("max_long_partial_prefills", 1)),
            "gpu_memory_utilization": float(evaluation.get("gpu_memory_utilization", 0.6)),
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
            "ignore_eos": bool(evaluation.get("ignore_eos", False)),
        },
        "phase1": {"baseline_mode": "chunked"},
    }
    (cmd, env) = build_eval_invocation(case_config, out_json_override=str(raw_path))
    env = apply_hf_resource_env(env, config)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=ROOT, env=env)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{case_name} failed with code {proc.returncode}; see {stdout_path} and {stderr_path}"
        )
    raw = _load_json(raw_path)
    return {
        "case": case_name,
        "fixed_token_budget": token_budget,
        "request_count": len(requests),
        "raw_path": str(raw_path.relative_to(run_root)),
        **_summarize_raw(raw),
    }


def _run_observation1(
    *,
    run_root: Path,
    model: dict[str, Any],
    evaluation: dict[str, Any],
    workload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    obs_cfg = dict(config.get("observation1") or {})
    budget = int(obs_cfg.get("fixed_token_budget", 768))
    rows = []
    for long_count in obs_cfg.get("long_counts", [1, 2]):
        count = int(long_count)
        case_name = "obs1_one_long" if count == 1 else f"obs1_{count}_long"
        rows.append(
            _run_fixed_baseline(
                run_root=run_root,
                case_name=case_name,
                model=model,
                evaluation=evaluation,
                token_budget=budget,
                requests=_build_trace(workload, count),
                config=config,
            )
        )
    return {
        "method_scope": "Fixed vLLM chunked baseline only; CUCUMIS/WaveSlice hooks are not run.",
        "metric": "Nearest-rank p99 TTFT measured from each request's scheduled arrival time over all short requests and repeats.",
        "fixed_token_budget": budget,
        "rows": rows,
    }


def _run_observation2(
    *,
    run_root: Path,
    model: dict[str, Any],
    evaluation: dict[str, Any],
    workload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    obs_cfg = dict(config.get("observation2") or {})
    long_count = int(obs_cfg.get("long_count", 2))
    trace = _build_trace(workload, long_count)
    rows = []
    for budget in obs_cfg.get("fixed_token_budgets", []):
        token_budget = int(budget)
        rows.append(
            _run_fixed_baseline(
                run_root=run_root,
                case_name=f"obs2_{long_count}_long_budget_{token_budget}",
                model=model,
                evaluation=evaluation,
                token_budget=token_budget,
                requests=trace,
                config=config,
            )
        )
    return {
        "method_scope": "Fixed vLLM chunked baseline only; CUCUMIS/WaveSlice hooks are not run.",
        "metric": "Nearest-rank p99 TTFT measured from each request's scheduled arrival time over all short requests and repeats.",
        "long_count": long_count,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/chapter2_observations_v2.json",
        help="Versioned JSON configuration for the rebuilt observations.",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help="Optional parent output directory; defaults to paths.out_root in the configuration.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run directory name; defaults to a timestamped rebuild name.",
    )
    parser.add_argument(
        "--observations",
        choices=["obs1", "obs2", "all"],
        default="all",
        help="Run the one-vs-two-long comparison, the ten-budget sweep, or both.",
    )
    args = parser.parse_args()
    config_path = resolve(ROOT, args.config)
    config = _load_json(config_path)
    out_parent = resolve(
        ROOT,
        args.out_root
        or str((config.get("paths") or {}).get("out_root", "results/chapter2_observations_v2")),
    )
    run_name = args.run_name or time.strftime("rebuild_%Y%m%d_%H%M%S")
    run_root = out_parent / run_name
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing observation run: {run_root}")
    run_root.mkdir(parents=True)
    model = _resolve_model(config)
    evaluation = dict(config.get("evaluation") or {})
    workload = dict(config.get("workload") or {})
    manifest = {
        "schema_version": "chapter2-observations-v2",
        "created_at_epoch_s": time.time(),
        "config_path": str(config_path),
        "config": config,
        "model": model,
        "method_scope": "Fixed vLLM chunked baseline only; CUCUMIS/WaveSlice hooks are not run.",
    }
    _write_json(run_root / "manifest.json", manifest)
    result: dict[str, Any] = {
        "schema_version": "chapter2-observations-v2",
        "model": model,
        "evaluation": evaluation,
        "workload": workload,
    }
    if args.observations in {"obs1", "all"}:
        result["observation1"] = _run_observation1(
            run_root=run_root, model=model, evaluation=evaluation, workload=workload, config=config
        )
    if args.observations in {"obs2", "all"}:
        result["observation2"] = _run_observation2(
            run_root=run_root, model=model, evaluation=evaluation, workload=workload, config=config
        )
    _write_json(run_root / "summary.json", result)
    print(f"[Saved] {run_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
