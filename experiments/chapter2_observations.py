from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_wave_slice")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from eval_support import Req, measure_input_tokens  # noqa: E402
from evaluate_waveslice_claims import _run_series  # noqa: E402
from experiments.chapter2_prestudy import (  # noqa: E402
    _ensure_dir,
    _load_json,
    _make_phase1_requests,
    _mean,
    _p99,
    _resolve_model,
    _safe_float,
    _write_json,
    _write_md,
)


def _to_reqs(items: list[dict[str, Any]]) -> list[Req]:
    return [
        Req(
            req_id=str(item["req_id"]),
            prompt=str(item["prompt"]),
            is_short=bool(item.get("is_short")),
            lora_tag=item.get("lora_tag"),
            arrival_offset_s=float(item.get("arrival_offset_s", 0.0) or 0.0),
        )
        for item in items
    ]


def _base_args(
    config: dict[str, Any],
    *,
    max_num_batched_tokens: int,
    warmup_iters: Optional[int],
    repeats: Optional[int],
    timeout_sec: Optional[int],
) -> Namespace:
    eval_cfg = dict(config.get("eval") or {})
    phase1_cfg = dict(config.get("phase1") or {})
    phase12_cfg = dict(config.get("phase12_soft_gate") or {})
    phase2_cfg = dict(config.get("phase2") or {})
    return Namespace(
        max_new_tokens=int(eval_cfg.get("max_new_tokens", 64)),
        ignore_eos=bool(eval_cfg.get("ignore_eos", False)),
        timeout_sec=int(timeout_sec if timeout_sec is not None else eval_cfg.get("timeout_sec", 240)),
        warmup_iters=int(warmup_iters if warmup_iters is not None else eval_cfg.get("warmup_iters", 1)),
        repeats=int(repeats if repeats is not None else eval_cfg.get("repeats", 2)),
        max_num_batched_tokens=int(max_num_batched_tokens),
        max_model_len=int((config.get("model") or {}).get("max_model_len") or eval_cfg.get("max_model_len", 3072)),
        gpu_memory_utilization=float(eval_cfg.get("gpu_memory_utilization", 0.60)),
        queue_reorder_mode=str(eval_cfg.get("queue_reorder_mode", "sjf")),
        queue_reorder_aging_quantum_us=float(eval_cfg.get("queue_reorder_aging_quantum_us", 20000.0)),
        phase1_objective_mode=str(phase1_cfg.get("objective_mode", "fair_escape")),
        phase1_gamma=float(phase1_cfg.get("gamma", 1.0)),
        phase1_ingress_target_chunk=int(phase1_cfg.get("ingress_target_chunk", 768)),
        phase1_ingress_direct_authoritative=bool(phase1_cfg.get("ingress_direct_authoritative", True)),
        phase1_ingress_exact_chunk=bool(phase1_cfg.get("ingress_exact_chunk", True)),
        phase1_force_min_chunk=int(phase1_cfg.get("force_min_chunk", 128)),
        phase1_target_long_fraction=float(phase1_cfg.get("target_long_fraction", 0.33)),
        phase1_runtime_adaptive_enabled=False,
        phase1_runtime_aggressive_long_fraction=0.33,
        phase1_runtime_conservative_long_fraction=0.50,
        phase1_runtime_aggressive_ingress_target_chunk=768,
        phase1_runtime_conservative_ingress_target_chunk=1536,
        phase1_runtime_queue_high_watermark=8,
        phase1_runtime_waiting_short_high_watermark=4,
        phase1_runtime_wait_us_high_watermark=1_000_000.0,
        phase1_runtime_long_high_watermark=3072,
        phase1_runtime_urgency_discount=0.55,
        phase1_runtime_ema_alpha=0.35,
        phase12_phase2_gate_mode=str(phase12_cfg.get("phase2_gate_mode", "soft")),
        phase12_phase2_soft_ratio_scale=float(phase12_cfg.get("soft_ratio_scale", 1.15)),
        phase12_phase2_soft_pressure_scale=float(phase12_cfg.get("soft_pressure_scale", 1.1)),
        phase12_phase2_soft_min_long_prefill=int(phase12_cfg.get("soft_min_long_prefill", 512)),
        phase12_phase2_soft_allow_mixed_decode=bool(phase12_cfg.get("soft_allow_mixed_decode", True)),
        phase12_phase2_soft_recent_strength_floor=float(phase12_cfg.get("soft_recent_strength_floor", 0.08)),
        phase12_phase2_soft_require_cashout_signal=bool(phase12_cfg.get("soft_require_cashout_signal", True)),
        phase12_phase2_soft_recent_chunk_match_scale=float(phase12_cfg.get("soft_recent_chunk_match_scale", 1.5)),
        phase12_phase2_soft_window_score_threshold=float(phase12_cfg.get("soft_window_score_threshold", 0.95)),
        phase12_phase2_soft_window_recent_weight=float(phase12_cfg.get("soft_window_recent_weight", 0.4)),
        phase12_phase2_soft_window_chunk_weight=float(phase12_cfg.get("soft_window_chunk_weight", 0.25)),
        phase12_phase2_soft_window_pressure_weight=float(phase12_cfg.get("soft_window_pressure_weight", 0.2)),
        phase12_phase2_soft_window_ratio_weight=float(phase12_cfg.get("soft_window_ratio_weight", 0.1)),
        phase12_phase2_soft_window_decode_bonus=float(phase12_cfg.get("soft_window_decode_bonus", 0.1)),
        phase2_dispatch_mode=str(phase2_cfg.get("dispatch_mode", "synchronized")),
        phase2_enable_mixed_prefill_decode=bool(phase2_cfg.get("enable_mixed_prefill_decode", False)),
        phase2_min_hetero_ratio=float(phase2_cfg.get("min_hetero_ratio", 4.0)),
        phase2_min_long_prefill=int(phase2_cfg.get("min_long_prefill", 768)),
        phase2_min_pressure_ratio=float(phase2_cfg.get("min_pressure_ratio", 4.0)),
        phase2_enable_scheduler_cashout=bool(phase2_cfg.get("enable_scheduler_cashout", True)),
        phase2_enable_execution_escape=bool(phase2_cfg.get("enable_execution_escape", True)),
        phase2_execution_escape_mode=str(phase2_cfg.get("execution_escape_mode", "bounded_spillover")),
        phase2_execution_escape_spillover_cap=int(phase2_cfg.get("execution_escape_spillover_cap", 3)),
        phase2_execution_escape_max_active=int(phase2_cfg.get("execution_escape_max_active", 5)),
        phase2_runtime_adaptive_enabled=False,
        phase2_runtime_low_pressure_min_hetero_ratio=6.0,
        phase2_runtime_high_pressure_min_hetero_ratio=4.0,
        phase2_runtime_low_pressure_min_pressure_ratio=6.0,
        phase2_runtime_high_pressure_min_pressure_ratio=4.0,
        phase2_runtime_low_pressure_min_long_prefill=1024,
        phase2_runtime_high_pressure_min_long_prefill=768,
        phase2_runtime_low_pressure_escape_spillover_cap=1,
        phase2_runtime_high_pressure_escape_spillover_cap=3,
        phase2_runtime_low_pressure_escape_max_active=2,
        phase2_runtime_high_pressure_escape_max_active=5,
        phase2_runtime_disable_execution_escape_below_pressure=-1.0,
        max_num_partial_prefills=int(eval_cfg.get("max_num_partial_prefills", 1)),
        max_long_partial_prefills=int(eval_cfg.get("max_long_partial_prefills", 1)),
        trust_remote_code=False,
    )


def _timing_values(rows: list[dict[str, Any]], req_filter) -> dict[str, Optional[float]]:
    first_ms: list[float] = []
    finish_ms: list[float] = []
    first_event: list[float] = []
    for row in rows:
        timings = row.get("request_timings") or {}
        if not isinstance(timings, dict):
            continue
        for req_id, item in timings.items():
            if not isinstance(item, dict) or not req_filter(str(req_id), item):
                continue
            first = _safe_float(item.get("scheduled_first_latency_ms"))
            if first is None:
                first = _safe_float(item.get("first_latency_ms"))
            finish = _safe_float(item.get("scheduled_finish_latency_ms"))
            if finish is None:
                finish = _safe_float(item.get("finish_latency_ms"))
            event = _safe_float(item.get("first_event_index"))
            if first is not None:
                first_ms.append(first)
            if finish is not None:
                finish_ms.append(finish)
            if event is not None:
                first_event.append(event)
    return {
        "first_p99_ms": _p99(first_ms),
        "first_mean_ms": _mean(first_ms),
        "finish_p99_ms": _p99(finish_ms),
        "finish_mean_ms": _mean(finish_ms),
        "first_event_mean": _mean(first_event),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    short_timings = _timing_values(rows, lambda _req_id, item: bool(item.get("is_short")))
    long_timings = _timing_values(rows, lambda _req_id, item: not bool(item.get("is_short")))
    return {
        "short_ttft_p99_ms": short_timings["first_p99_ms"],
        "round_wall_ms": _mean([_safe_float(row.get("round_wall_ms")) for row in rows]),
        "timed_out": any(bool(row.get("timed_out")) for row in rows),
        "finished_requests_mean": _mean([_safe_float(row.get("finished_requests")) for row in rows]),
        "total_requests_mean": _mean([_safe_float(row.get("total_requests")) for row in rows]),
        "short_timings": short_timings,
        "long_timings": long_timings,
    }


def _run_baseline(
    config: dict[str, Any],
    *,
    model: dict[str, Any],
    requests: list[dict[str, Any]],
    enable_chunked_prefill: bool,
    max_num_batched_tokens: int,
    warmup_iters: Optional[int],
    repeats: Optional[int],
    timeout_sec: Optional[int],
) -> list[dict[str, Any]]:
    args = _base_args(
        config,
        max_num_batched_tokens=max_num_batched_tokens,
        warmup_iters=warmup_iters,
        repeats=repeats,
        timeout_sec=timeout_sec,
    )
    return _run_series(
        args,
        model_name=str(model["model_name"]),
        model_path=str(model["model_path"]),
        reqs=_to_reqs(requests),
        enable_lora=False,
        mode="baseline",
        enable_chunked_prefill=enable_chunked_prefill,
    )


def _default_obs_config(config: dict[str, Any]) -> dict[str, Any]:
    obs_cfg = dict(config.get("chapter2_observations") or {})
    return {
        "chunk_tokens": int(obs_cfg.get("chunk_tokens", 768)),
        "chunk_sweep_tokens": list(obs_cfg.get("chunk_sweep_tokens") or [512, 768, 1536]),
        "short_count": int(obs_cfg.get("short_count", 6)),
        "short_prompt_repeat": int(obs_cfg.get("short_prompt_repeat", 2)),
        "long_prompt_repeat": int(obs_cfg.get("long_prompt_repeat", 70)),
        "short_start_s": float(obs_cfg.get("short_start_s", 0.10)),
        "short_gap_s": float(obs_cfg.get("short_gap_s", 0.04)),
        "sequential_delay_s": float(obs_cfg.get("sequential_delay_s", 6.0)),
    }


def _resolve_observation_model(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = dict(config.get("model") or {})
    if model_cfg and not any(key in model_cfg for key in ("key", "model_id", "lut_name")):
        compat_config = deepcopy(config)
        compat_config["model"] = {}
        model = _resolve_model(compat_config)
        if model_cfg.get("max_model_len") is not None:
            model["max_model_len"] = int(model_cfg["max_model_len"])
        return model
    return _resolve_model(config)


def _long_first_requests(obs_cfg: dict[str, Any], *, long_count: int = 1) -> list[dict[str, Any]]:
    base = _make_phase1_requests(
        pattern="long_first",
        short_count=int(obs_cfg["short_count"]),
        short_prompt_repeat=int(obs_cfg["short_prompt_repeat"]),
        long_prompt_repeat=int(obs_cfg["long_prompt_repeat"]),
        short_start_s=float(obs_cfg["short_start_s"]),
        short_gap_s=float(obs_cfg["short_gap_s"]),
        sequential_delay_s=float(obs_cfg["sequential_delay_s"]),
    )
    if long_count <= 1:
        return base
    long_prompt = str(next(item["prompt"] for item in base if not item.get("is_short")))
    shorts = [item for item in base if item.get("is_short")]
    longs = [
        {
            "req_id": f"long_{idx:02d}",
            "prompt": long_prompt,
            "is_short": False,
            "arrival_offset_s": idx * 0.01,
        }
        for idx in range(long_count)
    ]
    return longs + shorts


def _mixed_candidate_requests(obs_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = _long_first_requests(obs_cfg, long_count=1)
    long_req = next(item for item in base if not item.get("is_short"))
    short_prompt = str(next(item["prompt"] for item in base if item.get("is_short")))
    medium_prompt = short_prompt * 10
    late_prompt = short_prompt * 18
    return [
        dict(long_req),
        {
            "req_id": "tiny_waiting",
            "prompt": short_prompt,
            "is_short": True,
            "arrival_offset_s": float(obs_cfg["short_start_s"]),
        },
        {
            "req_id": "medium_waiting",
            "prompt": medium_prompt,
            "is_short": True,
            "arrival_offset_s": float(obs_cfg["short_start_s"]),
        },
        {
            "req_id": "medium_late",
            "prompt": late_prompt,
            "is_short": True,
            "arrival_offset_s": float(obs_cfg["short_start_s"]) + 0.18,
        },
    ]


def _plot_obs1(out_path: Path, summary: dict[str, Any]) -> None:
    labels = ["No chunking", "Fixed chunking"]
    ttft = [
        summary["no_chunking"]["short_ttft_p99_ms"],
        summary["fixed_chunking"]["short_ttft_p99_ms"],
    ]
    wall = [
        summary["no_chunking"]["round_wall_ms"],
        summary["fixed_chunking"]["round_wall_ms"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, values, title, ylabel in [
        (axes[0], ttft, "Short-request TTFT", "p99 TTFT (ms)"),
        (axes[1], wall, "Round wall time", "Wall time (ms)"),
    ]:
        ax.bar(range(2), values, color=["#6B7280", "#2563EB"], width=0.58)
        ax.set_xticks(range(2), labels, rotation=8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ymax = max(float(v or 0.0) for v in values) * 1.18
        ax.set_ylim(0, ymax if ymax > 0 else 1)
        for i, value in enumerate(values):
            if value is not None:
                ax.text(i, float(value) + ymax * 0.03, f"{float(value):.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _plot_obs2(out_path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(row["case"]) for row in rows]
    ttft = [row["short_ttft_p99_ms"] for row in rows]
    wall = [row["round_wall_ms"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    axes[0].bar(range(len(rows)), ttft, color="#2563EB", width=0.62)
    axes[0].set_ylabel("p99 TTFT (ms)")
    axes[0].set_title("Short-request TTFT")
    axes[1].bar(range(len(rows)), wall, color="#C2410C", width=0.62)
    axes[1].set_ylabel("Wall time (ms)")
    axes[1].set_title("Round wall time")
    for ax in axes:
        ax.set_xticks(range(len(rows)), labels, rotation=12, ha="right")
        ymax = max(float(v or 0.0) for v in (ttft if ax is axes[0] else wall)) * 1.18
        ax.set_ylim(0, ymax if ymax > 0 else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _plot_obs3(out_path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = [row["prefill_tokens"] for row in rows]
    ys = [row["boundary_fit_score"] for row in rows]
    labels = [row["req_id"] for row in rows]
    ax.scatter(xs, ys, s=70, color="#2563EB")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Request prefill tokens")
    ax.set_ylabel("Boundary-fit proxy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Boundary Value Depends on Candidate Fit")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def run_obs1(
    config: dict[str, Any],
    out_root: Path,
    *,
    model: dict[str, Any],
    warmup_iters: Optional[int],
    repeats: Optional[int],
    timeout_sec: Optional[int],
) -> dict[str, Any]:
    obs_cfg = _default_obs_config(config)
    exp_root = _ensure_dir(out_root / "obs1_long_prefill_delays_boundaries")
    requests = _long_first_requests(obs_cfg, long_count=1)
    max_len = int(model["max_model_len"])
    chunk_tokens = int(obs_cfg["chunk_tokens"])
    no_chunk_rows = _run_baseline(
        config,
        model=model,
        requests=requests,
        enable_chunked_prefill=False,
        max_num_batched_tokens=max_len,
        warmup_iters=warmup_iters,
        repeats=repeats,
        timeout_sec=timeout_sec,
    )
    fixed_rows = _run_baseline(
        config,
        model=model,
        requests=requests,
        enable_chunked_prefill=True,
        max_num_batched_tokens=chunk_tokens,
        warmup_iters=warmup_iters,
        repeats=repeats,
        timeout_sec=timeout_sec,
    )
    summary = {
        "claim": "Long prefills delay the next scheduler boundary; fixed chunking exposes earlier intervention points.",
        "method_scope": "baseline-only: no WaveSlice/CUCUMIS Phase I, Phase II, or beneficiary scheduling.",
        "chunk_tokens": chunk_tokens,
        "request_count": len(requests),
        "no_chunking": _summarize_rows(no_chunk_rows),
        "fixed_chunking": _summarize_rows(fixed_rows),
    }
    _write_json(exp_root / "raw_no_chunking.json", no_chunk_rows)
    _write_json(exp_root / "raw_fixed_chunking.json", fixed_rows)
    _write_json(exp_root / "summary.json", summary)
    _plot_obs1(exp_root / "obs1_chunking_creates_boundaries.png", summary)
    _write_md(
        exp_root / "summary.md",
        [
            "# Observation 1: Long Prefills Delay Scheduler Boundaries",
            "",
            f"- Scope: {summary['method_scope']}",
            f"- No chunking short TTFT p99 (ms): {summary['no_chunking']['short_ttft_p99_ms']}",
            f"- Fixed chunking short TTFT p99 (ms): {summary['fixed_chunking']['short_ttft_p99_ms']}",
            f"- No chunking wall time (ms): {summary['no_chunking']['round_wall_ms']}",
            f"- Fixed chunking wall time (ms): {summary['fixed_chunking']['round_wall_ms']}",
        ],
    )
    return summary


def run_obs2(
    config: dict[str, Any],
    out_root: Path,
    *,
    model: dict[str, Any],
    warmup_iters: Optional[int],
    repeats: Optional[int],
    timeout_sec: Optional[int],
) -> dict[str, Any]:
    obs_cfg = _default_obs_config(config)
    exp_root = _ensure_dir(out_root / "obs2_boundaries_do_not_guarantee_short_benefit")
    chunk_tokens = int(obs_cfg["chunk_tokens"])
    cases = [
        ("one_long", _long_first_requests(obs_cfg, long_count=1)),
        ("two_long_contenders", _long_first_requests(obs_cfg, long_count=2)),
    ]
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    for case_name, requests in cases:
        case_rows = _run_baseline(
            config,
            model=model,
            requests=requests,
            enable_chunked_prefill=True,
            max_num_batched_tokens=chunk_tokens,
            warmup_iters=warmup_iters,
            repeats=repeats,
            timeout_sec=timeout_sec,
        )
        raw[case_name] = case_rows
        case_summary = _summarize_rows(case_rows)
        rows.append({"case": case_name, "request_count": len(requests), **case_summary})

    for chunk in _default_obs_config(config)["chunk_sweep_tokens"]:
        case_name = f"two_long_chunk_{chunk}"
        case_rows = _run_baseline(
            config,
            model=model,
            requests=_long_first_requests(obs_cfg, long_count=2),
            enable_chunked_prefill=True,
            max_num_batched_tokens=int(chunk),
            warmup_iters=warmup_iters,
            repeats=repeats,
            timeout_sec=timeout_sec,
        )
        raw[case_name] = case_rows
        case_summary = _summarize_rows(case_rows)
        rows.append({"case": case_name, "request_count": 2 + int(obs_cfg["short_count"]), **case_summary})

    summary = {
        "claim": "More scheduler boundaries are opportunities, not guaranteed short-request relief.",
        "method_scope": "baseline-only fixed chunking; no WaveSlice/CUCUMIS Phase I, Phase II, or beneficiary scheduling.",
        "chunk_tokens": chunk_tokens,
        "rows": rows,
    }
    _write_json(exp_root / "raw.json", raw)
    _write_json(exp_root / "summary.json", summary)
    _plot_obs2(exp_root / "obs2_boundary_opportunity_not_benefit.png", rows)
    _write_md(
        exp_root / "summary.md",
        [
            "# Observation 2: More Boundaries Do Not Guarantee Short-Request Benefit",
            "",
            f"- Scope: {summary['method_scope']}",
            "",
            *[
                (
                    f"- {row['case']}: short_ttft_p99_ms={row['short_ttft_p99_ms']}, "
                    f"wall_ms={row['round_wall_ms']}, timed_out={row['timed_out']}"
                )
                for row in rows
            ],
        ],
    )
    return summary


def run_obs3(
    config: dict[str, Any],
    out_root: Path,
    *,
    model: dict[str, Any],
) -> dict[str, Any]:
    obs_cfg = _default_obs_config(config)
    exp_root = _ensure_dir(out_root / "obs3_boundary_value_is_request_dependent")
    requests = _mixed_candidate_requests(obs_cfg)
    tok_lens = measure_input_tokens(
        str(model["model_path"]),
        _to_reqs(requests),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
    )
    chunk_tokens = int(obs_cfg["chunk_tokens"])
    rows: list[dict[str, Any]] = []
    for item in requests:
        if not bool(item.get("is_short")):
            continue
        req_id = str(item["req_id"])
        prefill_tokens = int(tok_lens.get(req_id, 0))
        wait_s = max(0.0, float(obs_cfg["short_start_s"]) + 0.20 - float(item.get("arrival_offset_s", 0.0) or 0.0))
        fit_score = min(1.0, float(chunk_tokens) / max(1.0, float(prefill_tokens)))
        wait_score = min(1.0, wait_s / 0.20)
        rows.append(
            {
                "req_id": req_id,
                "arrival_offset_s": float(item.get("arrival_offset_s", 0.0) or 0.0),
                "prefill_tokens": prefill_tokens,
                "wait_proxy_s": wait_s,
                "size_fit_score": fit_score,
                "wait_score": wait_score,
                "boundary_fit_score": 0.60 * fit_score + 0.40 * wait_score,
            }
        )
    rows.sort(key=lambda row: row["boundary_fit_score"], reverse=True)
    summary = {
        "claim": "The same boundary has different value for different waiting requests.",
        "method_scope": "offline workload characterization only; no WaveSlice/CUCUMIS Phase I, Phase II, or beneficiary scorer.",
        "chunk_tokens": chunk_tokens,
        "rows": rows,
    }
    _write_json(exp_root / "requests.json", requests)
    _write_json(exp_root / "summary.json", summary)
    _plot_obs3(exp_root / "obs3_boundary_fit_proxy.png", rows)
    _write_md(
        exp_root / "summary.md",
        [
            "# Observation 3: Boundary Value Is Request-Dependent",
            "",
            f"- Scope: {summary['method_scope']}",
            "",
            *[
                (
                    f"- {row['req_id']}: tokens={row['prefill_tokens']}, "
                    f"wait_proxy_s={row['wait_proxy_s']:.3f}, "
                    f"boundary_fit_score={row['boundary_fit_score']:.3f}"
                )
                for row in rows
            ],
        ],
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline-only Chapter 2 observation suite.")
    parser.add_argument("experiment", choices=["all", "obs1", "obs2", "obs3"])
    parser.add_argument("--config", default="experiments/configs/chapter2_prestudy_v1.json")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=None)
    return parser.parse_args()


def _attach_existing_summaries(manifest: dict[str, Any], out_root: Path) -> None:
    existing = {
        "obs1": out_root / "obs1_long_prefill_delays_boundaries" / "summary.json",
        "obs2": out_root / "obs2_boundaries_do_not_guarantee_short_benefit" / "summary.json",
        "obs3": out_root / "obs3_boundary_value_is_request_dependent" / "summary.json",
    }
    for key, path in existing.items():
        if key in manifest or not path.exists():
            continue
        manifest[key] = _load_json(path)


def main() -> int:
    args = _parse_args()
    config = _load_json(Path(args.config))
    model = _resolve_observation_model(config)
    if bool(model.get("trust_remote_code", False)):
        raise RuntimeError("chapter2_observations currently runs only trust_remote_code=False models")
    out_root = Path(args.out_root or f"results/chapter2_observations_v1/{time.strftime('%Y%m%d_%H%M%S')}")
    _ensure_dir(out_root)
    manifest: dict[str, Any] = {
        "config_path": str(Path(args.config).resolve()),
        "out_root": str(out_root.resolve()),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "method_scope": "baseline-only observation suite; no CUCUMIS Phase I/Phase II result is used.",
    }
    if args.experiment in {"all", "obs1"}:
        manifest["obs1"] = run_obs1(
            config,
            out_root,
            model=model,
            warmup_iters=args.warmup_iters,
            repeats=args.repeats,
            timeout_sec=args.timeout_sec,
        )
    if args.experiment in {"all", "obs2"}:
        manifest["obs2"] = run_obs2(
            config,
            out_root,
            model=model,
            warmup_iters=args.warmup_iters,
            repeats=args.repeats,
            timeout_sec=args.timeout_sec,
        )
    if args.experiment in {"all", "obs3"}:
        manifest["obs3"] = run_obs3(config, out_root, model=model)
    _attach_existing_summaries(manifest, out_root)
    _write_json(out_root / "manifest.json", manifest)
    _write_md(
        out_root / "README.md",
        [
            "# Chapter 2 Baseline-Only Observations",
            "",
            "This directory is generated by `experiments/chapter2_observations.py`.",
            "",
            "Scope: baseline-only observations for Chapter 2. These runs do not use CUCUMIS Phase I, Phase II, or beneficiary scheduling results.",
            "",
            "Outputs:",
            "- `obs1_long_prefill_delays_boundaries/`",
            "- `obs2_boundaries_do_not_guarantee_short_benefit/`",
            "- `obs3_boundary_value_is_request_dependent/`",
        ],
    )
    print(f"[Chapter2-Obs] outputs written to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
