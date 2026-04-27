from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_wave_slice")

import matplotlib.pyplot as plt

from config.experiment_catalog import get_model_specs, safe_key
from experiments.local_resources import select_local_dataset_entries, select_local_model_entries
from experiments.model_assets import resolve_local_snapshot
from experiments.openworkload_models import ResolvedModel, resolve_model_entry


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _p99(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, int(round(0.99 * (len(ordered) - 1)))))
    return ordered[idx]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _text_token_estimate(text: str) -> int:
    return max(1, len(str(text or "").strip().split()))


def _length_bucket(tokens: float, *, short_leq: int, medium_leq: int) -> str:
    if tokens <= short_leq:
        return "short"
    if tokens <= medium_leq:
        return "medium"
    return "long"


def _beneficiary_short_fraction(reqs: list[dict[str, Any]]) -> Optional[float]:
    if not reqs:
        return None
    ordered = sorted(reqs, key=lambda item: float(item.get("arrival_offset_s", 0.0) or 0.0))
    prior_long = 0
    short_count = 0
    beneficiary = 0
    for item in ordered:
        is_short = bool(item.get("is_short"))
        if is_short:
            short_count += 1
            if prior_long > 0:
                beneficiary += 1
        else:
            prior_long += 1
    if short_count == 0:
        return None
    return beneficiary / short_count


def _arrival_gaps(reqs: list[dict[str, Any]]) -> list[float]:
    ordered = sorted(float(item.get("arrival_offset_s", 0.0) or 0.0) for item in reqs)
    return [ordered[i] - ordered[i - 1] for i in range(1, len(ordered))]


def _discover_latest_run(root: Path) -> Path:
    if root.is_file():
        return root
    if (root / "metadata").exists() and ((root / "workloads").exists() or (root / "raw").exists()):
        return root
    candidates = []
    for p in root.iterdir():
        if not p.is_dir() or not (p / "metadata").exists():
            continue
        score = 0
        if (p / "workloads").exists():
            score += 2
        if (p / "raw").exists():
            score += 2
        if (p / "metadata" / "suite_results.json").exists():
            score += 1
        candidates.append((score, p))
    if not candidates:
        raise FileNotFoundError(f"no run directory with metadata under {root}")
    candidates.sort(key=lambda item: (item[0], item[1].stat().st_mtime), reverse=True)
    return candidates[0][1]


def _fmt_metric(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _resolve_model(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(config.get("model") or {})
    if model_cfg:
        resolved = resolve_model_entry(model_cfg)
    else:
        resolved = resolve_model_entry("gemma-7b-it")
    local_snapshot = resolve_local_snapshot(resolved.model_id)
    if resolved.model_path_mode == "model_id":
        model_path = resolved.model_id
    elif resolved.model_path_mode == "local_snapshot_required":
        if not local_snapshot:
            raise FileNotFoundError(f"local snapshot required for {resolved.model_id}")
        model_path = local_snapshot
    else:
        model_path = local_snapshot or resolved.model_id
    max_model_len = model_cfg.get("max_model_len")
    if max_model_len is None:
        max_model_len = resolved.max_model_len_override or config.get("eval", {}).get("max_model_len", 3072)
    return {
        "key": resolved.key,
        "label": resolved.label,
        "model_name": resolved.lut_name,
        "model_id": resolved.model_id,
        "model_path": model_path,
        "trust_remote_code": bool(resolved.trust_remote_code),
        "max_model_len": int(max_model_len),
    }


def _phase1_prompt_short(idx: int, repeat: int) -> str:
    prefix = (
        "Interactive assistant turn. Keep the answer concise, direct, and faithful. "
        "A serving system may receive many such requests while longer document jobs are also active. "
    ) * max(1, repeat)
    return (
        f"{prefix}"
        f"Task {idx}: rewrite the sentence in one natural English sentence with the same meaning. "
        f"Sentence: 'Low latency matters most when the user is waiting for the next turn.'"
    )


def _phase1_prompt_long(repeat: int) -> str:
    passage = (
        "Long-context serving workloads mix summarization, long-document QA, and instruction following. "
        "When these requests share one GPU with short interactive turns, the scheduler must decide which request keeps making progress. "
    ) * max(1, repeat)
    return (
        f"{passage}"
        "Task: write exactly one sentence summarizing the passage without bullet points or headings."
    )


def _lora_prompt_short(idx: int, repeat: int) -> str:
    prefix = (
        "Short interactive request under adapter multiplexing. "
        "The answer should fit in one sentence. "
    ) * max(1, repeat)
    return (
        f"{prefix}"
        f"Task {idx}: translate this sentence into French and output only the translation. "
        "Sentence: 'Responsiveness should remain stable even when long jobs are active.'"
    )


def _lora_prompt_long(idx: int, repeat: int) -> str:
    passage = (
        "A long request occupies prefill and decode opportunities for many scheduling steps, "
        "while shorter tenant turns often need only a small amount of additional progress to complete. "
    ) * max(1, repeat)
    return (
        f"{passage}"
        f"Task {idx}: produce one sentence explaining the scheduling tension."
    )


def _make_phase1_requests(
    *,
    pattern: str,
    short_count: int,
    short_prompt_repeat: int,
    long_prompt_repeat: int,
    short_start_s: float,
    short_gap_s: float,
    sequential_delay_s: float,
) -> list[dict[str, Any]]:
    reqs: list[dict[str, Any]] = []
    long_prompt = _phase1_prompt_long(long_prompt_repeat)
    if pattern == "sequential":
        reqs.append({"req_id": "long_00", "prompt": long_prompt, "is_short": False, "arrival_offset_s": 0.0})
        for idx in range(short_count):
            reqs.append(
                {
                    "req_id": f"short_{idx:02d}",
                    "prompt": _phase1_prompt_short(idx, short_prompt_repeat),
                    "is_short": True,
                    "arrival_offset_s": sequential_delay_s + idx * short_gap_s,
                }
            )
        return reqs

    if pattern == "long_first":
        reqs.append({"req_id": "long_00", "prompt": long_prompt, "is_short": False, "arrival_offset_s": 0.0})
        for idx in range(short_count):
            reqs.append(
                {
                    "req_id": f"short_{idx:02d}",
                    "prompt": _phase1_prompt_short(idx, short_prompt_repeat),
                    "is_short": True,
                    "arrival_offset_s": short_start_s + idx * short_gap_s,
                }
            )
        return reqs

    if pattern == "short_first":
        for idx in range(short_count):
            reqs.append(
                {
                    "req_id": f"short_{idx:02d}",
                    "prompt": _phase1_prompt_short(idx, short_prompt_repeat),
                    "is_short": True,
                    "arrival_offset_s": idx * short_gap_s,
                }
            )
        reqs.append({"req_id": "long_00", "prompt": long_prompt, "is_short": False, "arrival_offset_s": short_start_s + short_count * short_gap_s})
        return reqs

    if pattern == "interleaved":
        reqs.append({"req_id": "long_00", "prompt": long_prompt, "is_short": False, "arrival_offset_s": 0.0})
        for idx in range(short_count):
            base_t = short_start_s + idx * short_gap_s
            reqs.append(
                {
                    "req_id": f"short_{idx:02d}",
                    "prompt": _phase1_prompt_short(idx, short_prompt_repeat),
                    "is_short": True,
                    "arrival_offset_s": base_t,
                }
            )
        return reqs

    raise ValueError(f"unknown phase1 pattern: {pattern}")


def _make_lora_requests(
    *,
    pattern: str,
    short_count: int,
    long_count: int,
    short_prompt_repeat: int,
    long_prompt_repeat: int,
    short_start_s: float,
    short_gap_s: float,
    long_gap_s: float,
    sequential_delay_s: float,
    mixed_adapters: bool,
) -> list[dict[str, Any]]:
    reqs: list[dict[str, Any]] = []

    def _tag(index: int) -> str:
        if not mixed_adapters:
            return "A"
        return "A" if index % 2 == 0 else "B"

    if pattern == "sequential":
        for idx in range(long_count):
            reqs.append(
                {
                    "req_id": f"long_{idx:02d}",
                    "prompt": _lora_prompt_long(idx, long_prompt_repeat),
                    "is_short": False,
                    "lora_tag": _tag(idx),
                    "arrival_offset_s": idx * long_gap_s,
                }
            )
        for idx in range(short_count):
            reqs.append(
                {
                    "req_id": f"short_{idx:02d}",
                    "prompt": _lora_prompt_short(idx, short_prompt_repeat),
                    "is_short": True,
                    "lora_tag": _tag(long_count + idx),
                    "arrival_offset_s": sequential_delay_s + idx * short_gap_s,
                }
            )
        return reqs

    if pattern == "long_first":
        for idx in range(long_count):
            reqs.append(
                {
                    "req_id": f"long_{idx:02d}",
                    "prompt": _lora_prompt_long(idx, long_prompt_repeat),
                    "is_short": False,
                    "lora_tag": _tag(idx),
                    "arrival_offset_s": idx * long_gap_s,
                }
            )
        for idx in range(short_count):
            reqs.append(
                {
                    "req_id": f"short_{idx:02d}",
                    "prompt": _lora_prompt_short(idx, short_prompt_repeat),
                    "is_short": True,
                    "lora_tag": _tag(long_count + idx),
                    "arrival_offset_s": short_start_s + idx * short_gap_s,
                }
            )
        return reqs

    if pattern == "short_first":
        for idx in range(short_count):
            reqs.append(
                {
                    "req_id": f"short_{idx:02d}",
                    "prompt": _lora_prompt_short(idx, short_prompt_repeat),
                    "is_short": True,
                    "lora_tag": _tag(idx),
                    "arrival_offset_s": idx * short_gap_s,
                }
            )
        for idx in range(long_count):
            reqs.append(
                {
                    "req_id": f"long_{idx:02d}",
                    "prompt": _lora_prompt_long(idx, long_prompt_repeat),
                    "is_short": False,
                    "lora_tag": _tag(short_count + idx),
                    "arrival_offset_s": short_start_s + short_count * short_gap_s + idx * long_gap_s,
                }
            )
        return reqs

    if pattern == "interleaved":
        cur_t = 0.0
        for idx in range(max(short_count, long_count)):
            if idx < long_count:
                reqs.append(
                    {
                        "req_id": f"long_{idx:02d}",
                        "prompt": _lora_prompt_long(idx, long_prompt_repeat),
                        "is_short": False,
                        "lora_tag": _tag(idx),
                        "arrival_offset_s": cur_t,
                    }
                )
                cur_t += long_gap_s
            if idx < short_count:
                reqs.append(
                    {
                        "req_id": f"short_{idx:02d}",
                        "prompt": _lora_prompt_short(idx, short_prompt_repeat),
                        "is_short": True,
                        "lora_tag": _tag(long_count + idx),
                        "arrival_offset_s": max(short_start_s, cur_t),
                    }
                )
                cur_t = max(short_start_s, cur_t) + short_gap_s
        return reqs

    raise ValueError(f"unknown lora pattern: {pattern}")


def _resolve_out_root(config: dict[str, Any], cli_out_root: Optional[str]) -> Path:
    if cli_out_root:
        return Path(cli_out_root)
    return Path(str((config.get("paths") or {}).get("out_root") or "results/chapter2_prestudy"))


def _run_eval_case(
    *,
    case_root: Path,
    case_name: str,
    model: dict[str, Any],
    eval_cfg: dict[str, Any],
    phase1_cfg: dict[str, Any],
    phase12_cfg: dict[str, Any],
    phase2_cfg: dict[str, Any],
    requests: list[dict[str, Any]],
    lora_requests: list[dict[str, Any]],
    include_phase12: bool,
    skip_phase2: bool,
    phase1_baseline_mode: str,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    workload_dir = _ensure_dir(case_root / "workloads")
    result_dir = _ensure_dir(case_root / "results")
    log_dir = _ensure_dir(case_root / "logs")
    req_path = workload_dir / f"{case_name}_requests.json"
    lora_req_path = workload_dir / f"{case_name}_lora_requests.json"
    out_json = result_dir / f"{case_name}_eval.json"
    stdout_path = log_dir / f"{case_name}.stdout.log"
    stderr_path = log_dir / f"{case_name}.stderr.log"
    _write_json(req_path, requests)
    _write_json(lora_req_path, lora_requests)

    cmd = [
        str(eval_cfg.get("python_bin") or sys.executable),
        "tests/evaluate_waveslice_claims.py",
        "--model-name",
        str(model["model_name"]),
        "--model-path",
        str(model["model_path"]),
        "--requests-json",
        str(req_path),
        "--lora-requests-json",
        str(lora_req_path),
        "--out-json",
        str(out_json),
        "--warmup-iters",
        str(int(eval_cfg.get("warmup_iters", 1))),
        "--repeats",
        str(int(eval_cfg.get("repeats", 2))),
        "--timeout-sec",
        str(int(eval_cfg.get("timeout_sec", 240))),
        "--max-new-tokens",
        str(int(eval_cfg.get("max_new_tokens", 64))),
        "--max-model-len",
        str(int(model["max_model_len"])),
        "--max-num-batched-tokens",
        str(int(eval_cfg.get("max_num_batched_tokens", 1536))),
        "--gpu-memory-utilization",
        str(float(eval_cfg.get("gpu_memory_utilization", 0.60))),
        "--queue-reorder-mode",
        str(eval_cfg.get("queue_reorder_mode", "sjf")),
        "--queue-reorder-aging-quantum-us",
        str(int(eval_cfg.get("queue_reorder_aging_quantum_us", 20000))),
        "--phase1-baseline-mode",
        str(phase1_baseline_mode),
        "--phase1-objective-mode",
        str(phase1_cfg.get("objective_mode", "fair_escape")),
        "--phase1-gamma",
        str(float(phase1_cfg.get("gamma", 1.0))),
        "--phase1-ingress-target-chunk",
        str(int(phase1_cfg.get("ingress_target_chunk", 768))),
        "--phase1-force-min-chunk",
        str(int(phase1_cfg.get("force_min_chunk", 128))),
        "--phase1-target-long-fraction",
        str(float(phase1_cfg.get("target_long_fraction", 0.33))),
        "--phase2-dispatch-mode",
        str(phase2_cfg.get("dispatch_mode", "synchronized")),
        "--phase2-min-hetero-ratio",
        str(float(phase2_cfg.get("min_hetero_ratio", 4.0))),
        "--phase2-min-long-prefill",
        str(int(phase2_cfg.get("min_long_prefill", 768))),
        "--phase2-min-pressure-ratio",
        str(float(phase2_cfg.get("min_pressure_ratio", 4.0))),
        "--phase2-execution-escape-mode",
        str(phase2_cfg.get("execution_escape_mode", "bounded_spillover")),
        "--phase2-execution-escape-spillover-cap",
        str(int(phase2_cfg.get("execution_escape_spillover_cap", 3))),
        "--phase2-execution-escape-max-active",
        str(int(phase2_cfg.get("execution_escape_max_active", 5))),
        "--phase12-phase2-gate-mode",
        str(phase12_cfg.get("phase2_gate_mode", "soft")),
        "--phase12-phase2-soft-ratio-scale",
        str(float(phase12_cfg.get("soft_ratio_scale", 1.15))),
        "--phase12-phase2-soft-pressure-scale",
        str(float(phase12_cfg.get("soft_pressure_scale", 1.1))),
        "--phase12-phase2-soft-min-long-prefill",
        str(int(phase12_cfg.get("soft_min_long_prefill", 512))),
        "--phase12-phase2-soft-recent-strength-floor",
        str(float(phase12_cfg.get("soft_recent_strength_floor", 0.08))),
        "--phase12-phase2-soft-recent-chunk-match-scale",
        str(float(phase12_cfg.get("soft_recent_chunk_match_scale", 1.5))),
        "--phase12-phase2-soft-window-score-threshold",
        str(float(phase12_cfg.get("soft_window_score_threshold", 0.95))),
        "--phase12-phase2-soft-window-recent-weight",
        str(float(phase12_cfg.get("soft_window_recent_weight", 0.4))),
        "--phase12-phase2-soft-window-chunk-weight",
        str(float(phase12_cfg.get("soft_window_chunk_weight", 0.25))),
        "--phase12-phase2-soft-window-pressure-weight",
        str(float(phase12_cfg.get("soft_window_pressure_weight", 0.2))),
        "--phase12-phase2-soft-window-ratio-weight",
        str(float(phase12_cfg.get("soft_window_ratio_weight", 0.1))),
        "--phase12-phase2-soft-window-decode-bonus",
        str(float(phase12_cfg.get("soft_window_decode_bonus", 0.1))),
    ]
    if ignore_eos:
        cmd.append("--ignore-eos")

    if bool(model.get("trust_remote_code", False)):
        cmd.append("--trust-remote-code")
    if bool(phase1_cfg.get("ingress_direct_authoritative", True)):
        cmd.append("--phase1-ingress-direct-authoritative")
    else:
        cmd.append("--no-phase1-ingress-direct-authoritative")
    if bool(phase1_cfg.get("ingress_exact_chunk", True)):
        cmd.append("--phase1-ingress-exact-chunk")
    if bool(phase12_cfg.get("soft_allow_mixed_decode", True)):
        cmd.append("--phase12-phase2-soft-allow-mixed-decode")
    else:
        cmd.append("--no-phase12-phase2-soft-allow-mixed-decode")
    if bool(phase12_cfg.get("soft_require_cashout_signal", True)):
        cmd.append("--phase12-phase2-soft-require-cashout-signal")
    else:
        cmd.append("--no-phase12-phase2-soft-require-cashout-signal")
    if bool(phase2_cfg.get("enable_mixed_prefill_decode", False)):
        cmd.append("--phase2-enable-mixed-prefill-decode")
    else:
        cmd.append("--no-phase2-enable-mixed-prefill-decode")
    if bool(phase2_cfg.get("enable_scheduler_cashout", True)):
        cmd.append("--phase2-enable-scheduler-cashout")
    else:
        cmd.append("--no-phase2-enable-scheduler-cashout")
    if bool(phase2_cfg.get("enable_execution_escape", True)):
        cmd.append("--phase2-enable-execution-escape")
    else:
        cmd.append("--no-phase2-enable-execution-escape")
    if include_phase12:
        cmd.append("--include-phase12")
    if skip_phase2:
        cmd.append("--skip-phase2")

    env = os.environ.copy()
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{case_name} failed with code {proc.returncode}. See {stdout_path} and {stderr_path}"
        )
    return _load_json(out_json)


def _summarize_request_timings(request_timings: dict[str, Any]) -> dict[str, Optional[float]]:
    short_ttft_ms: list[float] = []
    short_finish_ms: list[float] = []
    for item in request_timings.values():
        if not isinstance(item, dict) or not bool(item.get("is_short")):
            continue
        ttft = _safe_float(item.get("first_latency_ms"))
        finish = _safe_float(item.get("finish_latency_ms"))
        if ttft is not None:
            short_ttft_ms.append(ttft)
        if finish is not None:
            short_finish_ms.append(finish)
    return {
        "short_ttft_p99_ms": _p99(short_ttft_ms),
        "short_ttft_mean_ms": _mean(short_ttft_ms),
        "short_completion_p99_ms": _p99(short_finish_ms),
        "short_completion_mean_ms": _mean(short_finish_ms),
    }


def _short_completion_reference_by_req(rows: list[dict[str, Any]], timing_key: str) -> dict[str, float]:
    bucket: dict[str, list[float]] = {}
    for row in rows:
        timings = row.get(timing_key) or {}
        if not isinstance(timings, dict):
            continue
        for req_id, item in timings.items():
            if not isinstance(item, dict) or not bool(item.get("is_short")):
                continue
            finish = _safe_float(item.get("finish_latency_ms"))
            if finish is None or finish <= 0:
                continue
            bucket.setdefault(str(req_id), []).append(finish)
    return {req_id: float(sum(vals) / len(vals)) for req_id, vals in bucket.items() if vals}


def _row_short_completion_slowdown_p99(
    row: dict[str, Any],
    *,
    timing_key: str,
    reference_by_req: dict[str, float],
) -> Optional[float]:
    timings = row.get(timing_key) or {}
    if not isinstance(timings, dict):
        return None
    ratios: list[float] = []
    for req_id, item in timings.items():
        if not isinstance(item, dict) or not bool(item.get("is_short")):
            continue
        finish = _safe_float(item.get("finish_latency_ms"))
        ref = reference_by_req.get(str(req_id))
        if finish is None or ref is None or ref <= 0:
            continue
        ratios.append(finish / ref)
    return _p99(ratios)


def _mean_short_completion_slowdown(
    rows: list[dict[str, Any]],
    *,
    timing_key: str,
    reference_by_req: dict[str, float],
) -> Optional[float]:
    vals = [
        _row_short_completion_slowdown_p99(
            row,
            timing_key=timing_key,
            reference_by_req=reference_by_req,
        )
        for row in rows
    ]
    vals = [v for v in vals if v is not None]
    return _mean(vals)


def _aggregate_phase_rows(rows: list[dict[str, Any]], base_key: str, wave_key: str) -> dict[str, Any]:
    base_ttft = [_safe_float(row.get(base_key)) for row in rows]
    wave_ttft = [_safe_float(row.get(wave_key)) for row in rows]
    base_ttft = [v for v in base_ttft if v is not None]
    wave_ttft = [v for v in wave_ttft if v is not None]
    return {
        "base_ttft_p99_mean_ms": _mean(base_ttft),
        "wave_ttft_p99_mean_ms": _mean(wave_ttft),
    }


def _phase12_absolute_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return _phase_block_absolute_metrics(summary, phase_name="phase12")


def _phase2_absolute_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return _phase_block_absolute_metrics(summary, phase_name="phase2")


def _phase_block_absolute_metrics(summary: dict[str, Any], *, phase_name: str) -> dict[str, Any]:
    rows = list((summary.get("per_repeat") or {}).get(phase_name) or [])
    base_ttft = [_safe_float(row.get("base_ttft_short_p99_ms")) for row in rows]
    wave_ttft = [_safe_float(row.get("wave_ttft_short_p99_ms")) for row in rows]
    base_slow = [_safe_float(row.get("base_slowdown_short_p99")) for row in rows]
    wave_slow = [_safe_float(row.get("wave_slowdown_short_p99")) for row in rows]
    base_wall = [_safe_float(row.get("base_round_wall_ms")) for row in rows]
    wave_wall = [_safe_float(row.get("wave_round_wall_ms")) for row in rows]
    base_ttft = [v for v in base_ttft if v is not None]
    wave_ttft = [v for v in wave_ttft if v is not None]
    base_slow = [v for v in base_slow if v is not None]
    wave_slow = [v for v in wave_slow if v is not None]
    base_wall = [v for v in base_wall if v is not None]
    wave_wall = [v for v in wave_wall if v is not None]

    base_timing = [_summarize_request_timings(row.get("base_request_timings") or {}) for row in rows]
    wave_timing = [_summarize_request_timings(row.get("wave_request_timings") or {}) for row in rows]
    return {
        "rows": rows,
        "phase_name": phase_name,
        "baseline": {
            "short_ttft_p99_ms": _mean(base_ttft),
            "short_slowdown_p99": _mean(base_slow),
            "round_wall_ms": _mean(base_wall),
            "short_completion_p99_ms": _mean(
                [item["short_completion_p99_ms"] for item in base_timing if item["short_completion_p99_ms"] is not None]
            ),
        },
        "controlled": {
            "short_ttft_p99_ms": _mean(wave_ttft),
            "short_slowdown_p99": _mean(wave_slow),
            "round_wall_ms": _mean(wave_wall),
            "short_completion_p99_ms": _mean(
                [item["short_completion_p99_ms"] for item in wave_timing if item["short_completion_p99_ms"] is not None]
            ),
        },
    }


def _phase1_absolute_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    chunked_raw = (summary.get("phase1_baseline_chunked_raw") or {})
    no_chunk_raw = (summary.get("phase1_baseline_no_chunk_raw") or {})
    wave_rows = list((summary.get("per_repeat") or {}).get("phase1") or [])
    if not wave_rows:
        wave_rows = list(summary.get("phase1_rows") or [])
    no_chunk_control_rows = list((summary.get("per_repeat") or {}).get("phase1_chunked_vs_no_chunk") or [])
    if not no_chunk_control_rows:
        no_chunk_control_rows = list(summary.get("phase1_chunked_vs_no_chunk_rows") or [])
    wave_ttft = [_safe_float(row.get("wave_ttft_short_p99_ms")) for row in wave_rows]
    wave_wall = [_safe_float(row.get("wave_round_wall_ms")) for row in wave_rows]
    wave_ttft = [v for v in wave_ttft if v is not None]
    wave_wall = [v for v in wave_wall if v is not None]
    chunked_ttft_from_rows = [_safe_float(row.get("base_ttft_short_p99_ms")) for row in wave_rows]
    chunked_wall_from_rows = [_safe_float(row.get("base_round_wall_ms")) for row in wave_rows]
    chunked_ttft_from_rows = [v for v in chunked_ttft_from_rows if v is not None]
    chunked_wall_from_rows = [v for v in chunked_wall_from_rows if v is not None]
    no_chunk_ttft_from_rows = [_safe_float(row.get("base_ttft_short_p99_ms")) for row in no_chunk_control_rows]
    no_chunk_wall_from_rows = [_safe_float(row.get("base_round_wall_ms")) for row in no_chunk_control_rows]
    no_chunk_ttft_from_rows = [v for v in no_chunk_ttft_from_rows if v is not None]
    no_chunk_wall_from_rows = [v for v in no_chunk_wall_from_rows if v is not None]
    return {
        "no_chunk": {
            "short_ttft_p99_ms": _safe_float((no_chunk_raw.get("ttft_short_p99_ms") or {}).get("mean")) or _mean(no_chunk_ttft_from_rows),
            "round_wall_ms": _safe_float((no_chunk_raw.get("round_wall_ms") or {}).get("mean")) or _mean(no_chunk_wall_from_rows),
        },
        "fixed_chunking": {
            "short_ttft_p99_ms": _safe_float((chunked_raw.get("ttft_short_p99_ms") or {}).get("mean")) or _mean(chunked_ttft_from_rows),
            "round_wall_ms": _safe_float((chunked_raw.get("round_wall_ms") or {}).get("mean")) or _mean(chunked_wall_from_rows),
        },
        "online_control": {
            "short_ttft_p99_ms": _mean(wave_ttft),
            "round_wall_ms": _mean(wave_wall),
        },
        "wave_rows": wave_rows,
    }


def _pct_delta(new_value: Optional[float], ref_value: Optional[float]) -> Optional[float]:
    if new_value is None or ref_value is None or ref_value == 0:
        return None
    return (new_value - ref_value) / ref_value * 100.0


def _plot_bar_comparison(out_path: Path, title: str, ylabel: str, values: dict[str, Optional[float]]) -> None:
    items = [(label, value) for label, value in values.items() if value is not None]
    if not items:
        return
    labels = [label for label, _ in items]
    ys = [float(value) for _, value in items]
    x = list(range(len(items)))
    plt.figure(figsize=(7.2, 4.6))
    plt.bar(x, ys, color=["#5B8FF9", "#61DDAA", "#F6BD16"][: len(x)])
    plt.xticks(x, labels, rotation=12)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_e3_main_figure(
    out_path: Path,
    *,
    case_label: str,
    metrics: dict[str, Any],
) -> None:
    ttft_values = [
        metrics["no_chunk"]["short_ttft_p99_ms"],
        metrics["fixed_chunking"]["short_ttft_p99_ms"],
        metrics["online_control"]["short_ttft_p99_ms"],
    ]
    wall_values = [
        metrics["no_chunk"]["round_wall_ms"],
        metrics["fixed_chunking"]["round_wall_ms"],
        metrics["online_control"]["round_wall_ms"],
    ]
    if any(value is None for value in ttft_values + wall_values):
        return

    labels = ["No\nChunking", "Fixed\nChunking", "Latency-Aware\nAllocation"]
    colors = ["#E8684A", "#5B8FF9", "#61DDAA"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6))

    for ax, values, title, ylabel in [
        (axes[0], ttft_values, "Short-Request TTFT", "p99 TTFT (ms)"),
        (axes[1], wall_values, "Round Completion Time", "Wall Time (ms)"),
    ]:
        bars = ax.bar(range(3), values, color=colors, width=0.66)
        ax.set_xticks(range(3), labels)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis="both", labelsize=16)
        ymax = max(values) * 1.18
        ax.set_ylim(0, ymax)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + ymax * 0.02,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=16,
            )

    fig.suptitle("Chunking vs Latency-Aware Allocation", fontsize=15, y=0.935)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def _plot_timeline(out_path: Path, title: str, timings: dict[str, Any]) -> None:
    items = sorted(
        [
            (req_id, item)
            for req_id, item in timings.items()
            if isinstance(item, dict)
        ],
        key=lambda pair: float(pair[1].get("arrival_offset_s", 0.0) or 0.0),
    )
    if not items:
        return
    plt.figure(figsize=(8.0, max(3.2, 0.45 * len(items))))
    for idx, (req_id, item) in enumerate(items):
        start = float(item.get("arrival_offset_s", 0.0) or 0.0)
        finish = float(item.get("finish_latency_ms", 0.0) or 0.0) / 1000.0
        ttft = float(item.get("first_latency_ms", 0.0) or 0.0) / 1000.0
        is_short = bool(item.get("is_short"))
        color = "#61DDAA" if is_short else "#5B8FF9"
        plt.barh(idx, finish, left=start, color=color, alpha=0.85)
        plt.vlines(start + ttft, idx - 0.35, idx + 0.35, colors="#E8684A", linewidth=1.8)
    plt.yticks(range(len(items)), [req_id for req_id, _ in items])
    plt.xlabel("Time Since Round Start (s)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_e1_motivating_figure(
    out_path: Path,
    *,
    naive_timings: dict[str, Any],
    controlled_timings: dict[str, Any],
    max_short: int = 3,
) -> None:
    def _select_items(timings: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        items = [
            (req_id, item)
            for req_id, item in timings.items()
            if isinstance(item, dict)
        ]
        items.sort(key=lambda pair: float(pair[1].get("arrival_offset_s", 0.0) or 0.0))
        long_items = [(req_id, item) for req_id, item in items if not bool(item.get("is_short"))]
        short_items = [(req_id, item) for req_id, item in items if bool(item.get("is_short"))][:max_short]
        return long_items[:1] + short_items

    items = _select_items(naive_timings)
    if not items:
        return

    labels = ["L (long prompt)"] + [f"S{i} (short turn)" for i in range(1, len(items))]
    arrivals = [float(item.get("arrival_offset_s", 0.0) or 0.0) for _, item in items]
    arrivals[0] = 0.0

    # Stylized service allocations for a long-first arrival pattern.
    naive_segments = {
        labels[0]: [(0.00, 0.18), (0.18, 0.36), (0.36, 0.54)],
    }
    naive_completion = {labels[0]: 0.54}
    cur = 0.64
    for idx, label in enumerate(labels[1:], start=1):
        naive_segments[label] = [(cur, cur + 0.10)]
        naive_completion[label] = cur + 0.10
        cur += 0.10

    controlled_segments = {
        labels[0]: [(0.00, 0.18), (0.52, 0.72)],
    }
    controlled_completion = {labels[0]: 0.72}
    cur = 0.22
    for idx, label in enumerate(labels[1:], start=1):
        controlled_segments[label] = [(cur, cur + 0.10)]
        controlled_completion[label] = cur + 0.10
        cur += 0.10

    scenario_map = [
        ("Overlap Only", naive_segments, naive_completion, "Long prompt keeps most early service"),
        ("Overlap + Runtime Control", controlled_segments, controlled_completion, "Early service is redirected to short turns"),
    ]

    colors = {labels[0]: "#5B8FF9"}
    for label in labels[1:]:
        colors[label] = "#61DDAA"

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True, sharey=True)
    y_positions = list(range(len(labels)))

    for ax, (title, seg_map, completion_map, annotation) in zip(axes, scenario_map):
        for y, label, arrival in zip(y_positions, labels, arrivals):
            completion = completion_map[label]
            ax.hlines(y, arrival, completion, color="#B0B7C3", linewidth=3.0, alpha=0.85, zorder=1)
            for start_s, end_s in seg_map[label]:
                ax.broken_barh([(start_s, end_s - start_s)], (y - 0.26, 0.52), facecolors=colors[label], edgecolors="none", alpha=0.95, zorder=2)
            ax.plot(arrival, y, marker="v", color="#333333", markersize=7, zorder=3)
            ax.plot(completion, y, marker="o", color="#D9485F", markersize=5.5, zorder=3)
        ax.set_title(title, fontsize=16, pad=6)
        ax.text(0.98, 0.10, annotation, ha="right", va="bottom", transform=ax.transAxes, fontsize=12)
        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.30)
        ax.tick_params(axis="both", labelsize=13)

    axes[-1].set_xlabel("Time Since Long Arrival (s)", fontsize=16)
    axes[0].set_yticks(y_positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-0.02, 1.02)

    legend_handles = [
        plt.Line2D([0], [0], marker="v", color="#333333", lw=0, markersize=7),
        plt.Rectangle((0, 0), 1, 1, color="#5B8FF9", alpha=0.95),
        plt.Rectangle((0, 0), 1, 1, color="#61DDAA", alpha=0.95),
        plt.Line2D([0], [0], marker="o", color="#D9485F", lw=0, markersize=5.5),
    ]
    legend_labels = ["Arrival", "Service on long prompt", "Service on short turn", "Completion"]
    fig.legend(legend_handles, legend_labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=12)
    fig.suptitle("Overlap Does Not Determine Who Receives Early Service", fontsize=17, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)


def _plot_e5_relative_interference(
    out_path: Path,
    *,
    homogeneous_slowdown: Optional[float],
    mixed_slowdown: Optional[float],
) -> None:
    values = [homogeneous_slowdown, mixed_slowdown]
    if any(value is None for value in values):
        return

    labels = ["LoRA Homogeneous", "LoRA Mixed"]
    colors = ["#5B8FF9", "#F6BD16"]
    plt.figure(figsize=(6.6, 4.4))
    bars = plt.bar(range(2), values, color=colors, width=0.62)
    plt.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    plt.xticks(range(2), labels, rotation=10)
    plt.ylabel("Short Completion Slowdown\n(vs homogeneous LoRA)")
    plt.title("LoRA Multi-Tenant Relevance: Relative Interference")
    ymax = max(float(value) for value in values) * 1.10
    plt.ylim(0.96, max(1.04, ymax))
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + 0.004,
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def _collect_short_completion_ratios(
    rows: list[dict[str, Any]],
    *,
    timing_key: str,
    reference_by_req: dict[str, float],
) -> list[float]:
    ratios: list[float] = []
    for row in rows:
        timings = row.get(timing_key) or {}
        if not isinstance(timings, dict):
            continue
        for req_id, item in timings.items():
            if not isinstance(item, dict) or not bool(item.get("is_short")):
                continue
            finish = _safe_float(item.get("finish_latency_ms"))
            ref = reference_by_req.get(str(req_id))
            if finish is None or ref is None or ref <= 0:
                continue
            ratios.append(finish / ref)
    return ratios


def _plot_e5_distribution_figure(
    out_path: Path,
    *,
    homogeneous_ratios: list[float],
    mixed_ratios: list[float],
) -> None:
    if not homogeneous_ratios or not mixed_ratios:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    labels = ["Homogeneous", "Mixed"]
    colors = ["#5B8FF9", "#F6BD16"]

    bp = axes[0].boxplot(
        [homogeneous_ratios, mixed_ratios],
        labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.5)
    axes[0].axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Short Completion Slowdown\n(vs homogeneous LoRA)")
    axes[0].set_title("Distribution")

    def _cdf(values: list[float]) -> tuple[list[float], list[float]]:
        xs = sorted(float(v) for v in values)
        n = len(xs)
        ys = [(idx + 1) / n for idx in range(n)]
        return xs, ys

    h_x, h_y = _cdf(homogeneous_ratios)
    m_x, m_y = _cdf(mixed_ratios)
    axes[1].plot(h_x, h_y, color=colors[0], linewidth=2.0, label="Homogeneous")
    axes[1].plot(m_x, m_y, color=colors[1], linewidth=2.0, label="Mixed")
    axes[1].axvline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Short Completion Slowdown")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF")
    axes[1].legend(frameon=False)

    fig.suptitle("LoRA Multi-Tenancy Relevance: Relative Interference Distribution", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _collect_workload_records(run_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("workloads/*/*_requests.json")) + sorted(run_root.glob("workloads/*/*_lora_requests.json")):
        density = path.parent.name
        is_lora = path.name.endswith("_lora_requests.json")
        suffix = "_lora_requests.json" if is_lora else "_requests.json"
        model_key = path.name[: -len(suffix)]
        reqs = _load_json(path)
        for item in reqs:
            if not isinstance(item, dict):
                continue
            records.append(
                {
                    "density": density,
                    "model_key": model_key,
                    "path": str(path),
                    "is_lora": is_lora,
                    "req_id": str(item.get("req_id") or ""),
                    "is_short": bool(item.get("is_short")),
                    "arrival_offset_s": float(item.get("arrival_offset_s", 0.0) or 0.0),
                    "input_tokens": float(item.get("tokens") or _text_token_estimate(str(item.get("prompt") or ""))),
                    "source": str(item.get("source") or ""),
                    "lora_tag": str(item.get("lora_tag") or ""),
                }
            )
    return records


def _collect_output_records(run_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("raw/*/*_eval.json")):
        density = path.parent.name
        model_key = path.name[: -len("_dataset_eval.json")] if path.name.endswith("_dataset_eval.json") else path.stem
        data = _load_json(path)
        per_repeat = data.get("per_repeat") or {}
        for phase_name in ("phase2", "phase12"):
            rows = per_repeat.get(phase_name) or []
            for repeat_idx, row in enumerate(rows):
                texts = row.get("wave_texts") or {}
                for req_id, text in texts.items():
                    out.append(
                        {
                            "density": density,
                            "model_key": model_key,
                            "phase": phase_name,
                            "repeat_index": repeat_idx,
                            "req_id": str(req_id),
                            "output_tokens": float(_text_token_estimate(str(text or ""))),
                        }
                    )
    return out




def _experiment_root(out_root: Path, exp_name: str, config: dict[str, Any]) -> Path:
    model_key = str(config.get("_selected_model_key") or "").strip()
    if model_key:
        return _ensure_dir(out_root / exp_name / safe_key(model_key))
    return _ensure_dir(out_root / exp_name)


def _resolved_model_to_config_entry(model: ResolvedModel) -> dict[str, Any]:
    return {
        "key": model.key,
        "model_id": model.model_id,
        "lut_name": model.lut_name,
        "trust_remote_code": bool(model.trust_remote_code),
        "max_model_len_override": model.max_model_len_override,
        "model_path_mode": model.model_path_mode,
        "label": model.label,
        "reason": model.reason,
    }


def _config_for_model(config: dict[str, Any], model: ResolvedModel) -> dict[str, Any]:
    copied = deepcopy(config)
    copied["model"] = _resolved_model_to_config_entry(model)
    copied["_selected_model_key"] = model.key
    return copied


def _load_resource_catalog(config: dict[str, Any]) -> dict[str, Any]:
    catalog_path = str(config.get("resource_catalog_config") or "").strip()
    if not catalog_path:
        return {}
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"resource catalog config not found: {path}")
    return _load_json(path)


def _candidate_model_entries(config: dict[str, Any]) -> list[Any]:
    explicit = list(config.get("models") or [])
    if explicit:
        return explicit
    catalog = _load_resource_catalog(config)
    from_catalog = list(catalog.get("models") or [])
    if from_catalog:
        return from_catalog
    legacy_model = dict(config.get("model") or {})
    if legacy_model:
        return [legacy_model]
    return ["gemma-7b-it"]


def _candidate_dataset_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = list(config.get("datasets") or [])
    if explicit:
        return explicit
    catalog = _load_resource_catalog(config)
    return [dict(item) for item in (catalog.get("datasets") or []) if isinstance(item, dict)]


def _resolve_selected_models(config: dict[str, Any], model_keys_override: str) -> tuple[list[ResolvedModel], list[dict[str, Any]]]:
    selection_cfg = dict(config.get("resource_selection") or {})
    requested_keys = {item.strip() for item in model_keys_override.split(",") if item.strip()}
    candidate_entries = _candidate_model_entries(config)
    if requested_keys:
        filtered: list[Any] = []
        matched: set[str] = set()
        for entry in candidate_entries:
            resolved = resolve_model_entry(entry)
            if resolved.key in requested_keys:
                filtered.append(entry)
                matched.add(resolved.key)
        missing = sorted(requested_keys - matched)
        if missing:
            raise ValueError(f"unknown requested model keys: {missing}")
        candidate_entries = filtered

    mode = str(selection_cfg.get("model_mode") or ("local_all_runnable" if config.get("resource_catalog_config") else "configured")).strip().lower()
    if mode == "configured":
        models = [resolve_model_entry(entry) for entry in candidate_entries]
        diagnostics = [
            {
                "key": model.key,
                "model_id": model.model_id,
                "lut_name": model.lut_name,
                "label": model.label,
                "selected": True,
                "selection_mode": "configured",
            }
            for model in models
        ]
        return models, diagnostics
    if mode == "local_all_runnable":
        return select_local_model_entries(
            candidate_entries,
            require_runtime_sanity=bool(selection_cfg.get("require_runtime_sanity", True)),
            require_lora_support=bool(selection_cfg.get("require_lora_support", False)),
            exclude_name_substrings=list(selection_cfg.get("exclude_name_substrings") or []),
        )
    raise ValueError(f"unknown model selection mode: {mode}")


def _resolve_selected_datasets(config: dict[str, Any], dataset_keys_override: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selection_cfg = dict(config.get("resource_selection") or {})
    requested_keys = {item.strip() for item in dataset_keys_override.split(",") if item.strip()}
    candidate_entries = _candidate_dataset_entries(config)
    if requested_keys:
        filtered: list[dict[str, Any]] = []
        matched: set[str] = set()
        for entry in candidate_entries:
            key = str(entry.get("key") or "").strip()
            if key in requested_keys:
                filtered.append(entry)
                matched.add(key)
        missing = sorted(requested_keys - matched)
        if missing:
            raise ValueError(f"unknown requested dataset keys: {missing}")
        candidate_entries = filtered

    mode = str(selection_cfg.get("dataset_mode") or ("local_supported_from_catalog" if config.get("resource_catalog_config") else "configured")).strip().lower()
    if mode == "configured":
        diagnostics = [
            {
                "key": str(entry.get("key") or ""),
                "dataset_id": str(entry.get("dataset_id") or ""),
                "extractor": str(entry.get("extractor") or ""),
                "selected": True,
                "selection_mode": "configured",
            }
            for entry in candidate_entries
        ]
        return candidate_entries, diagnostics
    if mode == "local_supported_from_catalog":
        return select_local_dataset_entries(
            candidate_entries,
            require_supported_extractors=bool(selection_cfg.get("require_supported_extractors", True)),
        )
    raise ValueError(f"unknown dataset selection mode: {mode}")


def _run_model_matrix(
    *,
    exp_name: str,
    out_root: Path,
    base_config: dict[str, Any],
    models: list[ResolvedModel],
    runner,
) -> dict[str, Any]:
    exp_root = _ensure_dir(out_root / exp_name)
    per_model: dict[str, Any] = {}
    model_rows: list[dict[str, Any]] = []
    for index, model in enumerate(models, start=1):
        print(f"[Chapter2] {exp_name} ({index}/{len(models)}) model={model.label}", flush=True)
        summary = runner(_config_for_model(base_config, model), out_root)
        per_model[model.key] = summary
        model_rows.append(
            {
                "key": model.key,
                "label": model.label,
                "model_id": model.model_id,
                "lut_name": model.lut_name,
                "trust_remote_code": bool(model.trust_remote_code),
                "max_model_len_override": model.max_model_len_override,
            }
        )
    aggregate = {"models": model_rows, "per_model": per_model}
    _write_json(exp_root / "summary_all_models.json", aggregate)
    return aggregate


def _build_e4_suite_config(
    config: dict[str, Any],
    out_root: Path,
    selected_models: list[ResolvedModel],
    selected_datasets: list[dict[str, Any]],
) -> dict[str, Any]:
    catalog = _load_resource_catalog(config)
    e4_cfg = dict(config.get("e4_suite") or {})
    eval_cfg = deepcopy(catalog.get("eval") or {})
    eval_cfg.update(dict(config.get("eval") or {}))
    if e4_cfg.get("eval"):
        eval_cfg.update(dict(e4_cfg.get("eval") or {}))
    workload_cfg = deepcopy(e4_cfg.get("workload") or catalog.get("workload") or config.get("workload") or {})
    if not workload_cfg:
        raise ValueError("E4 requires workload settings from either resource_catalog_config or e4_suite.workload")
    scenario = deepcopy(e4_cfg.get("real_world_scenario") or catalog.get("real_world_scenario") or {})
    suite_name = str(e4_cfg.get("suite_name") or catalog.get("suite_name") or "chapter2_e4_density_suite")
    suite_out_root = out_root / "_e4_density_support" / "suite_runs"
    return {
        "suite_name": suite_name,
        "out_root": str(suite_out_root),
        "real_world_scenario": scenario,
        "models": [_resolved_model_to_config_entry(model) for model in selected_models],
        "optional_model_extensions": [],
        "datasets": selected_datasets,
        "optional_dataset_extensions": [],
        "workload": workload_cfg,
        "eval": eval_cfg,
        "phase1": deepcopy(config.get("phase1") or catalog.get("phase1") or {}),
        "phase12_soft_gate": deepcopy(config.get("phase12_soft_gate") or catalog.get("phase12_soft_gate") or {}),
        "phase2": deepcopy(config.get("phase2") or catalog.get("phase2") or {}),
    }


def _run_e4_density_suite(
    config: dict[str, Any],
    out_root: Path,
    selected_models: list[ResolvedModel],
    selected_datasets: list[dict[str, Any]],
) -> Path:
    support_root = _ensure_dir(out_root / "_e4_density_support")
    suite_cfg = _build_e4_suite_config(config, out_root, selected_models, selected_datasets)
    config_path = support_root / "generated_openworkload_config.json"
    stdout_path = support_root / "openworkload_suite.stdout.log"
    stderr_path = support_root / "openworkload_suite.stderr.log"
    _write_json(config_path, suite_cfg)
    cmd = [
        str((config.get("eval") or {}).get("python_bin") or sys.executable),
        "experiments/run_openworkload_execescape_suite.py",
        "--config",
        str(config_path),
        "--run-name",
        "suite",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"E4 density suite failed with code {proc.returncode}; see {stdout_path} and {stderr_path}"
        )
    return _discover_latest_run(Path(str(suite_cfg["out_root"])) / "suite")

def _run_e1_microbenchmark(config: dict[str, Any], out_root: Path) -> dict[str, Any]:
    model = _resolve_model(config)
    eval_cfg = dict(config.get("eval") or {})
    phase1_cfg = dict(config.get("phase1") or {})
    phase12_cfg = dict(config.get("phase12_soft_gate") or {})
    phase2_cfg = dict(config.get("phase2") or {})
    micro_cfg = dict(config.get("microbenchmark") or {})
    exp_root = _experiment_root(out_root, "E1_motivating_microbenchmark", config)

    shared_kwargs = {
        "short_count": int(micro_cfg.get("short_count", 6)),
        "long_count": int(micro_cfg.get("long_count", 1)),
        "short_prompt_repeat": int(micro_cfg.get("short_prompt_repeat", 2)),
        "long_prompt_repeat": int(micro_cfg.get("long_prompt_repeat", 40)),
        "short_start_s": float(micro_cfg.get("short_start_s", 0.20)),
        "short_gap_s": float(micro_cfg.get("short_gap_s", 0.05)),
        "long_gap_s": float(micro_cfg.get("long_gap_s", 0.0)),
        "sequential_delay_s": float(micro_cfg.get("sequential_delay_s", 6.0)),
    }

    sequential_lora = _make_lora_requests(pattern="sequential", mixed_adapters=True, **shared_kwargs)
    overlap_lora = _make_lora_requests(pattern="long_first", mixed_adapters=True, **shared_kwargs)
    phase1_stub = _make_phase1_requests(
        pattern="long_first",
        short_count=shared_kwargs["short_count"],
        short_prompt_repeat=shared_kwargs["short_prompt_repeat"],
        long_prompt_repeat=shared_kwargs["long_prompt_repeat"],
        short_start_s=shared_kwargs["short_start_s"],
        short_gap_s=shared_kwargs["short_gap_s"],
        sequential_delay_s=shared_kwargs["sequential_delay_s"],
    )

    sequential = _run_eval_case(
        case_root=exp_root,
        case_name="sequential_reference",
        model=model,
        eval_cfg=eval_cfg,
        phase1_cfg=phase1_cfg,
        phase12_cfg=phase12_cfg,
        phase2_cfg=phase2_cfg,
        requests=phase1_stub,
        lora_requests=sequential_lora,
        include_phase12=False,
        skip_phase2=False,
        phase1_baseline_mode="chunked",
    )
    overlap = _run_eval_case(
        case_root=exp_root,
        case_name="dynamic_overlap",
        model=model,
        eval_cfg=eval_cfg,
        phase1_cfg=phase1_cfg,
        phase12_cfg=phase12_cfg,
        phase2_cfg=phase2_cfg,
        requests=phase1_stub,
        lora_requests=overlap_lora,
        include_phase12=False,
        skip_phase2=False,
        phase1_baseline_mode="chunked",
    )

    seq_block = _phase2_absolute_metrics(sequential)
    overlap_block = _phase2_absolute_metrics(overlap)
    seq_metrics = dict(seq_block["baseline"])
    overlap_metrics = overlap_block
    sequential_ref = _short_completion_reference_by_req(seq_block["rows"], "base_request_timings")
    seq_metrics["short_slowdown_p99"] = 1.0 if sequential_ref else None
    overlap_metrics["baseline"]["short_slowdown_p99"] = _mean_short_completion_slowdown(
        overlap_block["rows"],
        timing_key="base_request_timings",
        reference_by_req=sequential_ref,
    )
    overlap_metrics["controlled"]["short_slowdown_p99"] = _mean_short_completion_slowdown(
        overlap_block["rows"],
        timing_key="wave_request_timings",
        reference_by_req=sequential_ref,
    )
    summary = {
        "sequential": seq_metrics,
        "naive_coexecution": overlap_metrics["baseline"],
        "controlled_progress": overlap_metrics["controlled"],
        "beneficiary_short_fraction": _beneficiary_short_fraction(overlap_lora),
    }
    _write_json(exp_root / "summary.json", summary)

    _plot_bar_comparison(
        exp_root / "short_ttft_p99_ms.png",
        "Short-Request TTFT Under Long-First Arrivals",
        "TTFT p99 (ms)",
        {
            "Sequential": summary["sequential"]["short_ttft_p99_ms"],
            "Naive Co-Exec": summary["naive_coexecution"]["short_ttft_p99_ms"],
            "Controlled": summary["controlled_progress"]["short_ttft_p99_ms"],
        },
    )
    _plot_bar_comparison(
        exp_root / "short_completion_p99_ms.png",
        "Short-Request Completion Time",
        "Completion p99 (ms)",
        {
            "Sequential": summary["sequential"]["short_completion_p99_ms"],
            "Naive Co-Exec": summary["naive_coexecution"]["short_completion_p99_ms"],
            "Controlled": summary["controlled_progress"]["short_completion_p99_ms"],
        },
    )
    _plot_bar_comparison(
        exp_root / "round_wall_ms.png",
        "Overall Wall Time",
        "Round Wall Time (ms)",
        {
            "Sequential": summary["sequential"]["round_wall_ms"],
            "Naive Co-Exec": summary["naive_coexecution"]["round_wall_ms"],
            "Controlled": summary["controlled_progress"]["round_wall_ms"],
        },
    )

    overlap_rows = list((overlap.get("per_repeat") or {}).get("phase2") or [])
    sequential_rows = list((sequential.get("per_repeat") or {}).get("phase2") or [])
    if sequential_rows:
        _plot_timeline(
            exp_root / "timeline_sequential.png",
            "Sequential Reference Timeline",
            sequential_rows[0].get("base_request_timings") or {},
        )
    if overlap_rows:
        _plot_timeline(
            exp_root / "timeline_naive.png",
            "Naive Co-Execution Timeline",
            overlap_rows[0].get("base_request_timings") or {},
        )
        _plot_timeline(
            exp_root / "timeline_controlled.png",
            "Controlled Progress Timeline",
            overlap_rows[0].get("wave_request_timings") or {},
        )
        _plot_e1_motivating_figure(
            exp_root / "motivating_progress_allocation.png",
            naive_timings=overlap_rows[0].get("base_request_timings") or {},
            controlled_timings=overlap_rows[0].get("wave_request_timings") or {},
        )

    md = [
        "# E1 Motivating Microbenchmark",
        "",
        f"- Beneficiary short fraction: {summary['beneficiary_short_fraction']}",
        f"- Sequential short TTFT p99 (ms): {summary['sequential']['short_ttft_p99_ms']}",
        f"- Naive co-exec short TTFT p99 (ms): {summary['naive_coexecution']['short_ttft_p99_ms']}",
        f"- Controlled short TTFT p99 (ms): {summary['controlled_progress']['short_ttft_p99_ms']}",
        f"- Sequential short slowdown p99: {summary['sequential']['short_slowdown_p99']}",
        f"- Naive co-exec short slowdown p99: {summary['naive_coexecution']['short_slowdown_p99']}",
        f"- Controlled short slowdown p99: {summary['controlled_progress']['short_slowdown_p99']}",
    ]
    _write_md(exp_root / "summary.md", md)
    return summary


def _run_e2_arrival_sensitivity(config: dict[str, Any], out_root: Path) -> dict[str, Any]:
    model = _resolve_model(config)
    eval_cfg = dict(config.get("eval") or {})
    phase1_cfg = dict(config.get("phase1") or {})
    phase12_cfg = dict(config.get("phase12_soft_gate") or {})
    phase2_cfg = dict(config.get("phase2") or {})
    sens_cfg = dict(config.get("arrival_sensitivity") or {})
    exp_root = _experiment_root(out_root, "E2_arrival_sensitivity", config)
    patterns = list(sens_cfg.get("patterns") or ["short_first", "interleaved", "long_first"])

    shared_kwargs = {
        "short_count": int(sens_cfg.get("short_count", 6)),
        "long_count": int(sens_cfg.get("long_count", 1)),
        "short_prompt_repeat": int(sens_cfg.get("short_prompt_repeat", 2)),
        "long_prompt_repeat": int(sens_cfg.get("long_prompt_repeat", 40)),
        "short_start_s": float(sens_cfg.get("short_start_s", 0.20)),
        "short_gap_s": float(sens_cfg.get("short_gap_s", 0.05)),
        "long_gap_s": float(sens_cfg.get("long_gap_s", 0.0)),
        "sequential_delay_s": float(sens_cfg.get("sequential_delay_s", 6.0)),
    }

    sequential_lora = _make_lora_requests(pattern="sequential", mixed_adapters=True, **shared_kwargs)
    phase1_stub = _make_phase1_requests(
        pattern="long_first",
        short_count=shared_kwargs["short_count"],
        short_prompt_repeat=shared_kwargs["short_prompt_repeat"],
        long_prompt_repeat=shared_kwargs["long_prompt_repeat"],
        short_start_s=shared_kwargs["short_start_s"],
        short_gap_s=shared_kwargs["short_gap_s"],
        sequential_delay_s=shared_kwargs["sequential_delay_s"],
    )
    seq_summary = _run_eval_case(
        case_root=exp_root,
        case_name="reference_sequential",
        model=model,
        eval_cfg=eval_cfg,
        phase1_cfg=phase1_cfg,
        phase12_cfg=phase12_cfg,
        phase2_cfg=phase2_cfg,
        requests=phase1_stub,
        lora_requests=sequential_lora,
        include_phase12=False,
        skip_phase2=False,
        phase1_baseline_mode="chunked",
    )
    seq_block = _phase2_absolute_metrics(seq_summary)
    seq_short_completion = seq_block["baseline"]["short_completion_p99_ms"]
    sequential_ref = _short_completion_reference_by_req(seq_block["rows"], "base_request_timings")

    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        lora_reqs = _make_lora_requests(pattern=pattern, mixed_adapters=True, **shared_kwargs)
        summary = _run_eval_case(
            case_root=exp_root,
            case_name=f"arrival_{pattern}",
            model=model,
            eval_cfg=eval_cfg,
            phase1_cfg=phase1_cfg,
            phase12_cfg=phase12_cfg,
            phase2_cfg=phase2_cfg,
            requests=phase1_stub,
            lora_requests=lora_reqs,
            include_phase12=False,
            skip_phase2=False,
            phase1_baseline_mode="chunked",
        )
        metrics = _phase2_absolute_metrics(summary)
        baseline_slowdown = _mean_short_completion_slowdown(
            metrics["rows"],
            timing_key="base_request_timings",
            reference_by_req=sequential_ref,
        )
        controlled_slowdown = _mean_short_completion_slowdown(
            metrics["rows"],
            timing_key="wave_request_timings",
            reference_by_req=sequential_ref,
        )
        base_completion = metrics["baseline"]["short_completion_p99_ms"]
        rows.append(
            {
                "pattern": pattern,
                "beneficiary_short_fraction": _beneficiary_short_fraction(lora_reqs),
                "baseline_short_ttft_p99_ms": metrics["baseline"]["short_ttft_p99_ms"],
                "baseline_short_slowdown_p99": baseline_slowdown,
                "baseline_short_completion_p99_ms": base_completion,
                "baseline_completion_slowdown_vs_sequential": (
                    (base_completion / seq_short_completion)
                    if base_completion is not None and seq_short_completion not in (None, 0)
                    else None
                ),
                "controlled_short_ttft_p99_ms": metrics["controlled"]["short_ttft_p99_ms"],
                "controlled_short_slowdown_p99": controlled_slowdown,
            }
        )

    summary = {"rows": rows, "sequential_reference_short_completion_p99_ms": seq_short_completion}
    _write_json(exp_root / "summary.json", summary)
    _plot_bar_comparison(
        exp_root / "arrival_baseline_ttft.png",
        "Arrival-Order Sensitivity of Short TTFT",
        "Baseline Short TTFT p99 (ms)",
        {row["pattern"]: row["baseline_short_ttft_p99_ms"] for row in rows},
    )
    _plot_bar_comparison(
        exp_root / "arrival_baseline_completion_slowdown.png",
        "Arrival-Order Sensitivity of Short Completion Slowdown",
        "Completion Slowdown vs Sequential",
        {row["pattern"]: row["baseline_completion_slowdown_vs_sequential"] for row in rows},
    )
    _write_md(
        exp_root / "summary.md",
        [
            "# E2 Arrival-Order Sensitivity",
            "",
            f"- Sequential reference short completion p99 (ms): {seq_short_completion}",
            "",
            "Pattern summary:",
            *[
                (
                    f"- {row['pattern']}: beneficiary={row['beneficiary_short_fraction']}, "
                    f"baseline_ttft={row['baseline_short_ttft_p99_ms']}, "
                    f"baseline_completion_slowdown={row['baseline_completion_slowdown_vs_sequential']}"
                )
                for row in rows
            ],
        ],
    )
    return summary


def _run_e3_chunking(config: dict[str, Any], out_root: Path) -> dict[str, Any]:
    model = _resolve_model(config)
    eval_cfg = dict(config.get("eval") or {})
    phase1_cfg = dict(config.get("phase1") or {})
    phase12_cfg = dict(config.get("phase12_soft_gate") or {})
    phase2_cfg = dict(config.get("phase2") or {})
    chunk_cfg = dict(config.get("chunking") or {})
    exp_root = _experiment_root(out_root, "E3_fixed_chunking_vs_online_control", config)

    reqs = _make_phase1_requests(
        pattern="long_first",
        short_count=int(chunk_cfg.get("short_count", 6)),
        short_prompt_repeat=int(chunk_cfg.get("short_prompt_repeat", 2)),
        long_prompt_repeat=int(chunk_cfg.get("long_prompt_repeat", 40)),
        short_start_s=float(chunk_cfg.get("short_start_s", 0.20)),
        short_gap_s=float(chunk_cfg.get("short_gap_s", 0.05)),
        sequential_delay_s=float(chunk_cfg.get("sequential_delay_s", 6.0)),
    )
    lora_stub = _make_lora_requests(
        pattern="long_first",
        short_count=2,
        long_count=1,
        short_prompt_repeat=1,
        long_prompt_repeat=4,
        short_start_s=0.0,
        short_gap_s=0.1,
        long_gap_s=0.0,
        sequential_delay_s=4.0,
        mixed_adapters=False,
    )
    summary = _run_eval_case(
        case_root=exp_root,
        case_name="chunking_tradeoff",
        model=model,
        eval_cfg=eval_cfg,
        phase1_cfg=phase1_cfg,
        phase12_cfg=phase12_cfg,
        phase2_cfg=phase2_cfg,
        requests=reqs,
        lora_requests=lora_stub,
        include_phase12=False,
        skip_phase2=True,
        phase1_baseline_mode="both",
    )
    metrics = _phase1_absolute_metrics(summary)
    _write_json(exp_root / "summary.json", metrics)
    _plot_bar_comparison(
        exp_root / "phase1_ttft.png",
        "No Chunking vs Fixed Chunking vs Online Control",
        "Short TTFT p99 (ms)",
        {
            "No Chunking": metrics["no_chunk"]["short_ttft_p99_ms"],
            "Fixed Chunking": metrics["fixed_chunking"]["short_ttft_p99_ms"],
            "Online Control": metrics["online_control"]["short_ttft_p99_ms"],
        },
    )
    _plot_bar_comparison(
        exp_root / "phase1_wall.png",
        "Round Wall Time Across Progress Controls",
        "Round Wall Time (ms)",
        {
            "No Chunking": metrics["no_chunk"]["round_wall_ms"],
            "Fixed Chunking": metrics["fixed_chunking"]["round_wall_ms"],
            "Online Control": metrics["online_control"]["round_wall_ms"],
        },
    )
    _write_md(
        exp_root / "summary.md",
        [
            "# E3 Fixed Chunking vs Online Control",
            "",
            f"- No chunking TTFT p99 (ms): {metrics['no_chunk']['short_ttft_p99_ms']}",
            f"- Fixed chunking TTFT p99 (ms): {metrics['fixed_chunking']['short_ttft_p99_ms']}",
            f"- Online control TTFT p99 (ms): {metrics['online_control']['short_ttft_p99_ms']}",
            f"- No chunking wall (ms): {metrics['no_chunk']['round_wall_ms']}",
            f"- Fixed chunking wall (ms): {metrics['fixed_chunking']['round_wall_ms']}",
            f"- Online control wall (ms): {metrics['online_control']['round_wall_ms']}",
        ],
    )
    return metrics


def _export_e3_paper_case(config: dict[str, Any], out_root: Path) -> dict[str, Any]:
    paper_cfg = dict(config.get("paper_e3") or {})
    source_result = Path(str(paper_cfg.get("source_result_json") or ""))
    source_meta = Path(str(paper_cfg.get("source_meta_json") or ""))
    if not source_result.exists():
        raise FileNotFoundError(f"E3 paper case source result not found: {source_result}")

    exp_root = _ensure_dir(out_root / "E3_fixed_chunking_vs_online_control_paper_case")
    data = _load_json(source_result)
    metrics = _phase1_absolute_metrics(data)
    case_label = str(paper_cfg.get("case_label") or source_result.stem)
    meta = _load_json(source_meta) if source_meta.exists() else {}

    ttft_chunk_vs_no = None
    ttft_online_vs_chunk = None
    wall_online_vs_chunk = None
    no_chunk_ttft = metrics["no_chunk"]["short_ttft_p99_ms"]
    chunk_ttft = metrics["fixed_chunking"]["short_ttft_p99_ms"]
    online_ttft = metrics["online_control"]["short_ttft_p99_ms"]
    chunk_wall = metrics["fixed_chunking"]["round_wall_ms"]
    online_wall = metrics["online_control"]["round_wall_ms"]
    if no_chunk_ttft and chunk_ttft:
        ttft_chunk_vs_no = no_chunk_ttft / chunk_ttft
    if chunk_ttft and online_ttft:
        ttft_online_vs_chunk = chunk_ttft / online_ttft
    if chunk_wall and online_wall:
        wall_online_vs_chunk = online_wall / chunk_wall

    summary = {
        "case_label": case_label,
        "source_result_json": str(source_result),
        "source_meta_json": str(source_meta) if source_meta.exists() else None,
        "workload_meta": meta,
        "metrics": metrics,
        "improvements": {
            "fixed_chunking_vs_no_chunk_ttft_ratio": ttft_chunk_vs_no,
            "online_vs_fixed_chunking_ttft_ratio": ttft_online_vs_chunk,
            "online_vs_fixed_chunking_wall_ratio": wall_online_vs_chunk,
            "fixed_chunking_ttft_reduction_vs_no_chunk_pct": (
                -_pct_delta(chunk_ttft, no_chunk_ttft) if chunk_ttft is not None and no_chunk_ttft is not None else None
            ),
            "online_ttft_reduction_vs_fixed_chunking_pct": (
                -_pct_delta(online_ttft, chunk_ttft) if online_ttft is not None and chunk_ttft is not None else None
            ),
            "online_wall_overhead_vs_fixed_chunking_pct": _pct_delta(online_wall, chunk_wall),
        },
    }
    _write_json(exp_root / "summary.json", summary)
    _plot_e3_main_figure(
        exp_root / "chunking_vs_online_control_main.png",
        case_label=case_label,
        metrics=metrics,
    )
    caption_lines = [
        "Figure E3. Fixed chunking reduces head-of-line blocking, but it remains a coarse proxy for runtime progress allocation.",
        (
            f"In this beneficiary-rich mixed workload ({case_label}), no chunking yields {no_chunk_ttft:.1f} ms short-request p99 TTFT, "
            f"fixed chunking reduces it to {chunk_ttft:.1f} ms, and online control further reduces it to {online_ttft:.1f} ms."
        ),
        (
            f"Relative to fixed chunking, online control lowers short-request p99 TTFT by "
            f"{summary['improvements']['online_ttft_reduction_vs_fixed_chunking_pct']:.1f}% "
            f"while changing round wall time by {summary['improvements']['online_wall_overhead_vs_fixed_chunking_pct']:.1f}%."
        ),
    ]
    _write_md(exp_root / "caption.md", caption_lines)
    _write_md(
        exp_root / "summary.md",
        [
            "# E3 Paper Case: Chunking vs Online Control",
            "",
            f"- Case label: {case_label}",
            f"- Source result: `{source_result}`",
            f"- Source meta: `{source_meta}`" if source_meta.exists() else "- Source meta: n/a",
            f"- No chunking short TTFT p99 (ms): {no_chunk_ttft}",
            f"- Fixed chunking short TTFT p99 (ms): {chunk_ttft}",
            f"- Online control short TTFT p99 (ms): {online_ttft}",
            f"- Fixed chunking wall time (ms): {chunk_wall}",
            f"- Online control wall time (ms): {online_wall}",
            (
                f"- Fixed chunking TTFT reduction vs no chunking (%): "
                f"{summary['improvements']['fixed_chunking_ttft_reduction_vs_no_chunk_pct']}"
            ),
            (
                f"- Online TTFT reduction vs fixed chunking (%): "
                f"{summary['improvements']['online_ttft_reduction_vs_fixed_chunking_pct']}"
            ),
            (
                f"- Online wall overhead vs fixed chunking (%): "
                f"{summary['improvements']['online_wall_overhead_vs_fixed_chunking_pct']}"
            ),
        ],
    )
    return summary


def _run_e4_density(
    config: dict[str, Any],
    out_root: Path,
    cli_suite_root: Optional[str],
    selected_models: Optional[list[ResolvedModel]] = None,
    selected_datasets: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    paths_cfg = dict(config.get("paths") or {})
    suite_override = str(cli_suite_root or paths_cfg.get("density_suite_root") or "").strip()
    if suite_override:
        suite_root = _discover_latest_run(Path(suite_override))
    else:
        if not selected_models:
            raise ValueError("E4 requires selected models when no suite root override is provided")
        if not selected_datasets:
            raise ValueError("E4 requires selected datasets when no suite root override is provided")
        suite_root = _run_e4_density_suite(config, out_root, selected_models, selected_datasets)
    exp_root = _ensure_dir(out_root / "E4_density_sweep")
    rows: list[dict[str, Any]] = []
    for result_path in sorted(suite_root.glob("raw/*/*_eval.json")):
        density = result_path.parent.name
        data = _load_json(result_path)
        phase12_rows = list((data.get("per_repeat") or {}).get("phase12") or [])
        if not phase12_rows:
            continue
        base_ttft = [_safe_float(row.get("base_ttft_short_p99_ms")) for row in phase12_rows]
        wave_ttft = [_safe_float(row.get("wave_ttft_short_p99_ms")) for row in phase12_rows]
        base_slow = [_safe_float(row.get("base_slowdown_short_p99")) for row in phase12_rows]
        wave_slow = [_safe_float(row.get("wave_slowdown_short_p99")) for row in phase12_rows]
        base_ttft = [v for v in base_ttft if v is not None]
        wave_ttft = [v for v in wave_ttft if v is not None]
        base_slow = [v for v in base_slow if v is not None]
        wave_slow = [v for v in wave_slow if v is not None]
        rows.append(
            {
                "density": density,
                "model": result_path.name,
                "baseline_short_ttft_p99_ms": _mean(base_ttft),
                "controlled_short_ttft_p99_ms": _mean(wave_ttft),
                "baseline_short_slowdown_p99": _mean(base_slow),
                "controlled_short_slowdown_p99": _mean(wave_slow),
            }
        )

    density_order = ["low", "mid", "high", "peak"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["density"]), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for density in sorted(grouped.keys(), key=lambda item: density_order.index(item) if item in density_order else item):
        bucket = grouped[density]
        summary_rows.append(
            {
                "density": density,
                "baseline_short_ttft_p99_ms": _mean([float(item["baseline_short_ttft_p99_ms"]) for item in bucket if item["baseline_short_ttft_p99_ms"] is not None]),
                "controlled_short_ttft_p99_ms": _mean([float(item["controlled_short_ttft_p99_ms"]) for item in bucket if item["controlled_short_ttft_p99_ms"] is not None]),
                "baseline_short_slowdown_p99": _mean([float(item["baseline_short_slowdown_p99"]) for item in bucket if item["baseline_short_slowdown_p99"] is not None]),
                "controlled_short_slowdown_p99": _mean([float(item["controlled_short_slowdown_p99"]) for item in bucket if item["controlled_short_slowdown_p99"] is not None]),
            }
        )
    summary = {"suite_root": str(suite_root), "rows": summary_rows}
    _write_json(exp_root / "summary.json", summary)

    if summary_rows:
        x = list(range(len(summary_rows)))
        labels = [row["density"] for row in summary_rows]
        plt.figure(figsize=(6.8, 4.0))
        plt.plot(x, [row["baseline_short_ttft_p99_ms"] for row in summary_rows], marker="o", markersize=8, linewidth=2.8, label="Baseline")
        plt.plot(x, [row["controlled_short_ttft_p99_ms"] for row in summary_rows], marker="o", markersize=8, linewidth=2.8, label="Controlled")
        plt.xticks(x, labels, fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel("Short TTFT p99 (ms)", fontsize=18)
        plt.title("Density Sweep: Short TTFT Inflation", fontsize=20)
        plt.legend(frameon=False, fontsize=16)
        plt.tight_layout()
        plt.savefig(exp_root / "density_ttft_curve.png", dpi=280)
        plt.close()

        improve_ratio = []
        for row in summary_rows:
            baseline = row["baseline_short_ttft_p99_ms"]
            controlled = row["controlled_short_ttft_p99_ms"]
            if baseline in (None, 0) or controlled is None:
                improve_ratio.append(None)
            else:
                improve_ratio.append((float(baseline) - float(controlled)) / float(baseline))

        improve_points = [value for value in improve_ratio if value is not None]
        if improve_points:
            plt.figure(figsize=(7.2, 4.6))
            plt.plot(
                x,
                improve_ratio,
                marker="o",
                color="#61DDAA",
                linewidth=2.0,
            )
            plt.xticks(x, labels)
            plt.ylabel("Relative TTFT Reduction")
            plt.title("Density Sweep: Relative Short-TTFT Improvement")
            plt.ylim(0, max(0.1, max(improve_points) * 1.12))
            plt.tight_layout()
            plt.savefig(exp_root / "density_ttft_reduction_ratio.png", dpi=180)
            plt.close()

        plt.figure(figsize=(7.2, 4.6))
        plt.plot(x, [row["baseline_short_slowdown_p99"] for row in summary_rows], marker="o", label="Baseline")
        plt.plot(x, [row["controlled_short_slowdown_p99"] for row in summary_rows], marker="o", label="Controlled")
        plt.xticks(x, labels)
        plt.ylabel("Short Slowdown p99")
        plt.title("Density Sweep: Interference Severity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(exp_root / "density_slowdown_curve.png", dpi=180)
        plt.close()

    _write_md(
        exp_root / "summary.md",
        [
            "# E4 Density Sweep",
            "",
            f"- Suite root: `{suite_root}`",
            "",
            *[
                (
                    f"- {row['density']}: baseline_ttft={row['baseline_short_ttft_p99_ms']}, "
                    f"controlled_ttft={row['controlled_short_ttft_p99_ms']}, "
                    f"baseline_slowdown={row['baseline_short_slowdown_p99']}, "
                    f"controlled_slowdown={row['controlled_short_slowdown_p99']}"
                )
                for row in summary_rows
            ],
        ],
    )
    return summary


def _run_e5_lora(config: dict[str, Any], out_root: Path) -> dict[str, Any]:
    model = _resolve_model(config)
    eval_cfg = dict(config.get("eval") or {})
    phase1_cfg = dict(config.get("phase1") or {})
    phase12_cfg = dict(config.get("phase12_soft_gate") or {})
    phase2_cfg = dict(config.get("phase2") or {})
    lora_cfg = dict(config.get("lora_relevance") or {})
    exp_root = _experiment_root(out_root, "E5_lora_multitenancy_relevance", config)

    shared_kwargs = {
        "short_count": int(lora_cfg.get("short_count", 6)),
        "long_count": int(lora_cfg.get("long_count", 2)),
        "short_prompt_repeat": int(lora_cfg.get("short_prompt_repeat", 2)),
        "long_prompt_repeat": int(lora_cfg.get("long_prompt_repeat", 36)),
        "short_start_s": float(lora_cfg.get("short_start_s", 0.20)),
        "short_gap_s": float(lora_cfg.get("short_gap_s", 0.05)),
        "long_gap_s": float(lora_cfg.get("long_gap_s", 0.02)),
        "sequential_delay_s": float(lora_cfg.get("sequential_delay_s", 6.0)),
    }
    phase1_ref = _make_phase1_requests(
        pattern="long_first",
        short_count=shared_kwargs["short_count"],
        short_prompt_repeat=shared_kwargs["short_prompt_repeat"],
        long_prompt_repeat=shared_kwargs["long_prompt_repeat"],
        short_start_s=shared_kwargs["short_start_s"],
        short_gap_s=shared_kwargs["short_gap_s"],
        sequential_delay_s=shared_kwargs["sequential_delay_s"],
    )
    lora_stub = _make_lora_requests(pattern="long_first", mixed_adapters=True, **shared_kwargs)
    non_lora_summary = _run_eval_case(
        case_root=exp_root,
        case_name="non_lora_reference",
        model=model,
        eval_cfg=eval_cfg,
        phase1_cfg=phase1_cfg,
        phase12_cfg=phase12_cfg,
        phase2_cfg=phase2_cfg,
        requests=phase1_ref,
        lora_requests=lora_stub,
        include_phase12=False,
        skip_phase2=True,
        phase1_baseline_mode="chunked",
    )
    homogeneous_summary = _run_eval_case(
        case_root=exp_root,
        case_name="lora_homogeneous",
        model=model,
        eval_cfg=eval_cfg,
        phase1_cfg=phase1_cfg,
        phase12_cfg=phase12_cfg,
        phase2_cfg=phase2_cfg,
        requests=phase1_ref,
        lora_requests=_make_lora_requests(pattern="long_first", mixed_adapters=False, **shared_kwargs),
        include_phase12=False,
        skip_phase2=False,
        phase1_baseline_mode="chunked",
    )
    mixed_summary = _run_eval_case(
        case_root=exp_root,
        case_name="lora_mixed_adapters",
        model=model,
        eval_cfg=eval_cfg,
        phase1_cfg=phase1_cfg,
        phase12_cfg=phase12_cfg,
        phase2_cfg=phase2_cfg,
        requests=phase1_ref,
        lora_requests=_make_lora_requests(pattern="long_first", mixed_adapters=True, **shared_kwargs),
        include_phase12=False,
        skip_phase2=False,
        phase1_baseline_mode="chunked",
    )

    non_lora_metrics = _phase1_absolute_metrics(non_lora_summary)
    homogeneous_block = _phase2_absolute_metrics(homogeneous_summary)
    mixed_block = _phase2_absolute_metrics(mixed_summary)
    homogeneous_metrics = homogeneous_block["baseline"]
    mixed_metrics = mixed_block["baseline"]
    lora_ref = _short_completion_reference_by_req(homogeneous_block["rows"], "base_request_timings")
    homogeneous_metrics["short_slowdown_p99"] = 1.0 if lora_ref else None
    mixed_metrics["short_slowdown_p99"] = _mean_short_completion_slowdown(
        mixed_block["rows"],
        timing_key="base_request_timings",
        reference_by_req=lora_ref,
    )
    homogeneous_ratio_distribution = _collect_short_completion_ratios(
        homogeneous_block["rows"],
        timing_key="base_request_timings",
        reference_by_req=lora_ref,
    )
    mixed_ratio_distribution = _collect_short_completion_ratios(
        mixed_block["rows"],
        timing_key="base_request_timings",
        reference_by_req=lora_ref,
    )
    summary = {
        "non_lora_reference": non_lora_metrics["fixed_chunking"],
        "lora_homogeneous": homogeneous_metrics,
        "lora_mixed_adapters": mixed_metrics,
        "lora_homogeneous_ratio_distribution": homogeneous_ratio_distribution,
        "lora_mixed_ratio_distribution": mixed_ratio_distribution,
    }
    _write_json(exp_root / "summary.json", summary)
    _plot_bar_comparison(
        exp_root / "lora_latency_dispersion.png",
        "LoRA Multi-Tenancy Relevance Check",
        "Short TTFT p99 (ms)",
        {
            "Non-LoRA": summary["non_lora_reference"]["short_ttft_p99_ms"],
            "LoRA Homogeneous": summary["lora_homogeneous"]["short_ttft_p99_ms"],
            "LoRA Mixed": summary["lora_mixed_adapters"]["short_ttft_p99_ms"],
        },
    )
    _plot_e5_relative_interference(
        exp_root / "lora_relative_interference.png",
        homogeneous_slowdown=summary["lora_homogeneous"]["short_slowdown_p99"],
        mixed_slowdown=summary["lora_mixed_adapters"]["short_slowdown_p99"],
    )
    _plot_e5_distribution_figure(
        exp_root / "lora_relative_interference_distribution.png",
        homogeneous_ratios=homogeneous_ratio_distribution,
        mixed_ratios=mixed_ratio_distribution,
    )
    _write_md(
        exp_root / "summary.md",
        [
            "# E5 LoRA Multi-Tenancy Relevance Check",
            "",
            f"- Non-LoRA short TTFT p99 (ms): {summary['non_lora_reference']['short_ttft_p99_ms']}",
            f"- LoRA homogeneous short TTFT p99 (ms): {summary['lora_homogeneous']['short_ttft_p99_ms']}",
            f"- LoRA mixed short TTFT p99 (ms): {summary['lora_mixed_adapters']['short_ttft_p99_ms']}",
            f"- LoRA homogeneous slowdown p99: {summary['lora_homogeneous']['short_slowdown_p99']}",
            f"- LoRA mixed slowdown p99: {summary['lora_mixed_adapters']['short_slowdown_p99']}",
            f"- Homogeneous slowdown samples: {homogeneous_ratio_distribution}",
            f"- Mixed slowdown samples: {mixed_ratio_distribution}",
        ],
    )
    return summary



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter-2 prestudy runner for the v1 branch (public entrypoints: E1-E5).")
    parser.add_argument(
        "experiment",
        choices=[
            "all",
            "e1",
            "e2",
            "e3",
            "e3paper",
            "e4",
            "e5",
        ],
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/chapter2_prestudy_v1.json",
        help="JSON config describing paths, model/data selection, and runtime defaults.",
    )
    parser.add_argument("--out-root", default="", help="Override output root.")
    parser.add_argument("--density-suite-root", default="", help="Override E4 density-suite root.")
    parser.add_argument("--model-keys", default="", help="Optional comma-separated model keys to restrict the selected model set.")
    parser.add_argument("--dataset-keys", default="", help="Optional comma-separated dataset keys to restrict the selected dataset set.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = _load_json(Path(args.config))
    out_root = _resolve_out_root(config, args.out_root or None)
    _ensure_dir(out_root)
    selected_models, model_diagnostics = _resolve_selected_models(config, args.model_keys or "")
    selected_datasets, dataset_diagnostics = _resolve_selected_datasets(config, args.dataset_keys or "")
    if args.experiment in {"all", "e1", "e2", "e3", "e4", "e5"} and not selected_models:
        raise RuntimeError("no models selected for Chapter 2 prestudy")
    manifest: dict[str, Any] = {
        "config_path": str(Path(args.config).resolve()),
        "out_root": str(out_root.resolve()),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_models": [
            {
                "key": model.key,
                "label": model.label,
                "model_id": model.model_id,
                "lut_name": model.lut_name,
                "trust_remote_code": bool(model.trust_remote_code),
                "max_model_len_override": model.max_model_len_override,
            }
            for model in selected_models
        ],
        "selected_datasets": selected_datasets,
        "model_selection_diagnostics": model_diagnostics,
        "dataset_selection_diagnostics": dataset_diagnostics,
    }

    if args.experiment in {"all", "e1"}:
        manifest["e1"] = _run_model_matrix(
            exp_name="E1_motivating_microbenchmark",
            out_root=out_root,
            base_config=config,
            models=selected_models,
            runner=_run_e1_microbenchmark,
        )
    if args.experiment in {"all", "e2"}:
        manifest["e2"] = _run_model_matrix(
            exp_name="E2_arrival_sensitivity",
            out_root=out_root,
            base_config=config,
            models=selected_models,
            runner=_run_e2_arrival_sensitivity,
        )
    if args.experiment in {"all", "e3"}:
        manifest["e3"] = _run_model_matrix(
            exp_name="E3_fixed_chunking_vs_online_control",
            out_root=out_root,
            base_config=config,
            models=selected_models,
            runner=_run_e3_chunking,
        )
    if args.experiment in {"all", "e3paper"}:
        manifest["e3paper"] = _export_e3_paper_case(config, out_root)
    if args.experiment in {"all", "e4"}:
        manifest["e4"] = _run_e4_density(
            config,
            out_root,
            args.density_suite_root or None,
            selected_models,
            selected_datasets,
        )
    if args.experiment in {"all", "e5"}:
        manifest["e5"] = _run_model_matrix(
            exp_name="E5_lora_multitenancy_relevance",
            out_root=out_root,
            base_config=config,
            models=selected_models,
            runner=_run_e5_lora,
        )

    _write_json(out_root / "manifest.json", manifest)
    print(f"[Chapter2] outputs written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
