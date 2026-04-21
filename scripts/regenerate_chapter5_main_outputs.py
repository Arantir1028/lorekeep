from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUCUMIS_FIG_DIR = ROOT / "results" / "chapter5_exports" / "default"
DEFAULT_MAIN_RUN = ROOT / "results" / "openworkload_v1_local_realworld_lora8" / "20260420_chapter5_eq1_eq5_lora8"
DEFAULT_BASELINE_RUN = ROOT / "results" / "chapter5_baseline_variants" / "20260421_chapter5_baseline_variants_lora7_v2"
CUCUMIS_FIG_DIR = DEFAULT_CUCUMIS_FIG_DIR
MAIN_RUN = DEFAULT_MAIN_RUN
BASELINE_RUN = DEFAULT_BASELINE_RUN
DENSITY_ORDER = ["low", "mid", "high", "peak"]
COMPARABLE_METHODS = ["Continuous batching", "Chunked prefill", "Sarathi", "CUCUMIS"]
ABLATION_METHODS = ["Chunked prefill", "CUCUMIS-II", "CUCUMIS-Strict", "CUCUMIS"]
LONG_COST_METHODS = ["Continuous batching", "Chunked prefill", "Sarathi", "CUCUMIS-II", "CUCUMIS", "CUCUMIS-Strict"]
METHOD_DISPLAY = {
    "Continuous batching": "vLLM",
    "Chunked prefill": "Chunked prefill",
    "Sarathi": "Sarathi",
    "CUCUMIS-I": "CUCUMIS-I",
    "CUCUMIS-II": "CUCUMIS-II",
    "CUCUMIS": "CUCUMIS",
    "CUCUMIS-Strict": "CUCUMIS-Strict",
}
METHOD_COLORS = {
    "Continuous batching": "#4C78A8",
    "Chunked prefill": "#F58518",
    "Sarathi": "#72B7B2",
    "CUCUMIS-I": "#54A24B",
    "CUCUMIS-II": "#4C78A8",
    "CUCUMIS": "#E45756",
    "CUCUMIS-Strict": "#54A24B",
}
MODEL_LABELS = {
    "mistral-7b-v0.1": "Mistral-7B-v0.1",
    "mistral-7b-instruct-v0.2": "Mistral-7B-Instruct-v0.2",
    "zephyr-7b-beta": "Zephyr-7B-Beta",
    "openchat-3.5-0106": "OpenChat-3.5-0106",
    "gemma-7b-it": "Gemma-7B-IT",
    "baichuan2-7b-chat": "Baichuan2-7B-Chat",
    "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct",
}
MODEL_ORDER = list(MODEL_LABELS.values())
MODEL_SHORT_LABELS = {
    "Mistral-7B-v0.1": "Mistral",
    "Mistral-7B-Instruct-v0.2": "M-I",
    "Zephyr-7B-Beta": "Zephyr",
    "OpenChat-3.5-0106": "OpenChat",
    "Gemma-7B-IT": "Gemma",
    "Baichuan2-7B-Chat": "BC2",
    "Qwen2.5-7B-Instruct": "Qwen",
    "All": "All",
}


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_paper_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 15,
        }
    )


def _annotate_bar_values(
    ax: plt.Axes,
    xs: list[int],
    means: list[float],
    errs: list[float],
    *,
    fmt: str,
) -> None:
    ymax = ax.get_ylim()[1]
    lift = ymax * 0.02
    for x, mean_val, err_val in zip(xs, means, errs):
        ax.text(
            x,
            mean_val + err_val + lift,
            format(mean_val, fmt),
            ha="center",
            va="bottom",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.3},
            zorder=5,
        )


def _display_method(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def _display_model(model: str) -> str:
    return MODEL_SHORT_LABELS.get(model, model)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("percentile on empty sequence")
    if q <= 0:
        return ordered[0]
    if q >= 100:
        return ordered[-1]
    k = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ordered[lo]
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _safe_p99(hook_report: dict[str, object], key: str) -> float | None:
    block = (hook_report or {}).get(key)
    if not isinstance(block, dict):
        return None
    val = block.get("p99")
    return float(val) if isinstance(val, (int, float)) else None


def _request_count(request_timings: dict[str, object]) -> int:
    return sum(1 for item in request_timings.values() if isinstance(item, dict))


def _long_finish_p99_ms(request_timings: dict[str, object]) -> float | None:
    values = [
        float(item["finish_latency_ms"])
        for item in request_timings.values()
        if isinstance(item, dict)
        and not bool(item.get("is_short"))
        and isinstance(item.get("finish_latency_ms"), (int, float))
    ]
    if not values:
        return None
    return _percentile(values, 99.0)


def _finish_p99_ms(request_timings: dict[str, object], *, is_short: bool | None = None) -> float | None:
    values = [
        float(item["finish_latency_ms"])
        for item in request_timings.values()
        if isinstance(item, dict)
        and isinstance(item.get("finish_latency_ms"), (int, float))
        and (is_short is None or bool(item.get("is_short")) == is_short)
    ]
    if not values:
        return None
    return _percentile(values, 99.0)


def _ttft_p99_ms(request_timings: dict[str, object], *, is_short: bool | None = None) -> float:
    values = [
        float(item["first_latency_ms"])
        for item in request_timings.values()
        if isinstance(item, dict)
        and isinstance(item.get("first_latency_ms"), (int, float))
        and (is_short is None or bool(item.get("is_short")) == is_short)
    ]
    return _percentile(values, 99.0)


def _mean(values: list[float]) -> float:
    return float(mean(float(v) for v in values))


def _mean_ci95(values: list[float]) -> tuple[float, float]:
    data = [float(v) for v in values]
    if not data:
        raise ValueError("empty values")
    if len(data) == 1:
        return data[0], 0.0
    return float(mean(data)), float(1.96 * stdev(data) / math.sqrt(len(data)))


def _method_row(
    *,
    method: str,
    model_key: str,
    density: str,
    repeat_index: int,
    hook_report: dict[str, object],
    request_timings: dict[str, object],
    round_wall_ms: float,
    mismatch_rate: float | None = None,
    baseline_noise_floor: float | None = None,
    incremental_mismatch_rate: float | None = None,
    apply_ratio: float | None = None,
) -> dict[str, object]:
    request_count = _request_count(request_timings)
    return {
        "method": method,
        "model": MODEL_LABELS.get(model_key, model_key),
        "density": density,
        "repeat_index": int(repeat_index),
        "all_ttft_p99_ms": _safe_p99(hook_report, "ttft_ms_all"),
        "short_ttft_p99_ms": _safe_p99(hook_report, "ttft_ms_short"),
        "long_ttft_p99_ms": _safe_p99(hook_report, "ttft_ms_long"),
        "all_finish_p99_ms": _finish_p99_ms(request_timings),
        "all_slowdown_p99": _safe_p99(hook_report, "slowdown_all"),
        "short_slowdown_p99": _safe_p99(hook_report, "slowdown_short"),
        "long_slowdown_p99": _safe_p99(hook_report, "slowdown_long"),
        "long_finish_p99_ms": _long_finish_p99_ms(request_timings),
        "round_wall_ms": float(round_wall_ms),
        "completed_requests_per_s": (request_count * 1000.0 / float(round_wall_ms)) if round_wall_ms else None,
        "mismatch_rate": mismatch_rate,
        "baseline_noise_floor": baseline_noise_floor,
        "incremental_mismatch_rate": incremental_mismatch_rate,
        "apply_ratio": apply_ratio,
    }


def _collect_baseline_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for density in DENSITY_ORDER:
        raw_dir = BASELINE_RUN / "raw" / density
        for result_path in sorted(raw_dir.glob("*.json")):
            payload = _read_json(result_path)
            if not isinstance(payload, dict):
                continue
            per_repeat = dict(payload.get("per_repeat") or {})
            name = result_path.name
            if name.endswith("_fixed_chunk_vs_sarathi.json"):
                model_key = name[: -len("_fixed_chunk_vs_sarathi.json")]
                for rep in list(per_repeat.get("phase2") or []):
                    if not isinstance(rep, dict):
                        continue
                    repeat_index = int(rep.get("repeat_index") or 0)
                    rows.append(
                        _method_row(
                            method="Chunked prefill",
                            model_key=model_key,
                            density=density,
                            repeat_index=repeat_index,
                            hook_report=dict(rep.get("base_hook_report") or {}),
                            request_timings=dict(rep.get("base_request_timings") or {}),
                            round_wall_ms=float(rep.get("base_round_wall_ms") or 0.0),
                            baseline_noise_floor=float(rep.get("baseline_noise_error_rate") or 0.0),
                        )
                    )
                    rows.append(
                        _method_row(
                            method="Sarathi",
                            model_key=model_key,
                            density=density,
                            repeat_index=repeat_index,
                            hook_report=dict(rep.get("wave_hook_report") or {}),
                            request_timings=dict(rep.get("wave_request_timings") or {}),
                            round_wall_ms=float(rep.get("wave_round_wall_ms") or 0.0),
                            mismatch_rate=float(rep.get("wave_error_rate") or 0.0),
                            baseline_noise_floor=float(rep.get("baseline_noise_error_rate") or 0.0),
                            incremental_mismatch_rate=float(rep.get("incremental_error_rate") or 0.0),
                            apply_ratio=float(rep.get("phase2_apply_ratio") or 0.0),
                        )
                    )
            elif name.endswith("_strict_no_chunk.json"):
                model_key = name[: -len("_strict_no_chunk.json")]
                for rep in list(per_repeat.get("phase2") or []):
                    if not isinstance(rep, dict):
                        continue
                    repeat_index = int(rep.get("repeat_index") or 0)
                    rows.append(
                        _method_row(
                            method="Continuous batching",
                            model_key=model_key,
                            density=density,
                            repeat_index=repeat_index,
                            hook_report=dict(rep.get("base_hook_report") or {}),
                            request_timings=dict(rep.get("base_request_timings") or {}),
                            round_wall_ms=float(rep.get("base_round_wall_ms") or 0.0),
                            baseline_noise_floor=float(rep.get("baseline_noise_error_rate") or 0.0),
                        )
                    )
                    rows.append(
                        _method_row(
                            method="CUCUMIS-II",
                            model_key=model_key,
                            density=density,
                            repeat_index=repeat_index,
                            hook_report=dict(rep.get("wave_hook_report") or {}),
                            request_timings=dict(rep.get("wave_request_timings") or {}),
                            round_wall_ms=float(rep.get("wave_round_wall_ms") or 0.0),
                            mismatch_rate=float(rep.get("wave_error_rate") or 0.0),
                            baseline_noise_floor=float(rep.get("baseline_noise_error_rate") or 0.0),
                            incremental_mismatch_rate=float(rep.get("incremental_error_rate") or 0.0),
                            apply_ratio=float(rep.get("phase2_apply_ratio") or 0.0),
                        )
                    )
                for rep in list(per_repeat.get("phase2_strict") or []):
                    if not isinstance(rep, dict):
                        continue
                    rows.append(
                        _method_row(
                            method="CUCUMIS-Strict",
                            model_key=model_key,
                            density=density,
                            repeat_index=int(rep.get("repeat_index") or 0),
                            hook_report=dict(rep.get("strict_hook_report") or {}),
                            request_timings=dict(rep.get("strict_request_timings") or {}),
                            round_wall_ms=float(rep.get("strict_round_wall_ms") or 0.0),
                            mismatch_rate=float(rep.get("strict_error_rate") or 0.0),
                            baseline_noise_floor=float(rep.get("baseline_noise_error_rate") or 0.0),
                            incremental_mismatch_rate=float(rep.get("incremental_error_rate") or 0.0),
                            apply_ratio=float(rep.get("strict_apply_ratio") or 0.0),
                        )
                    )
    return rows


def _collect_main_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    suite_rows = _read_json(MAIN_RUN / "metadata" / "suite_results.json")
    if not isinstance(suite_rows, list):
        return rows
    ok_cases = [
        row
        for row in suite_rows
        if isinstance(row, dict) and str(row.get("status")) == "ok" and str(row.get("model_key")) != "gemma-2-9b-it"
    ]
    for case in ok_cases:
        model_key = str(case.get("model_key") or "")
        density = str(case.get("density") or "")
        result_path = MAIN_RUN / "raw" / density / f"{model_key}_dataset_eval.json"
        payload = _read_json(result_path)
        if not isinstance(payload, dict):
            continue
        per_repeat = dict(payload.get("per_repeat") or {})
        for repeat_index, rep in enumerate(list(per_repeat.get("phase1") or [])):
            if not isinstance(rep, dict):
                continue
            rows.append(
                _method_row(
                    method="CUCUMIS-I",
                    model_key=model_key,
                    density=density,
                    repeat_index=int(rep.get("repeat_index") or repeat_index),
                    hook_report=dict(rep.get("wave_hook_report") or {}),
                    request_timings=dict(rep.get("wave_request_timings") or {}),
                    round_wall_ms=float(rep.get("wave_round_wall_ms") or 0.0),
                    mismatch_rate=float(rep.get("error_rate") or 0.0),
                    baseline_noise_floor=float(rep.get("baseline_noise_error_rate") or 0.0),
                    incremental_mismatch_rate=float(rep.get("incremental_error_rate") or 0.0),
                    apply_ratio=float(rep.get("scheduler_apply_ratio") or 0.0),
                )
            )
        for rep in list(per_repeat.get("phase12") or []):
            if not isinstance(rep, dict):
                continue
            rows.append(
                _method_row(
                    method="CUCUMIS",
                    model_key=model_key,
                    density=density,
                    repeat_index=int(rep.get("repeat_index") or 0),
                    hook_report=dict(rep.get("wave_hook_report") or {}),
                    request_timings=dict(rep.get("wave_request_timings") or {}),
                    round_wall_ms=float(rep.get("wave_round_wall_ms") or 0.0),
                    mismatch_rate=float(rep.get("wave_error_rate") or 0.0),
                    baseline_noise_floor=float(rep.get("baseline_noise_error_rate") or 0.0),
                    incremental_mismatch_rate=float(rep.get("incremental_error_rate") or 0.0),
                    apply_ratio=float(rep.get("phase2_apply_ratio") or 0.0),
                )
            )
    return rows


def _case_mean_values(
    rows: list[dict[str, object]], *, key: str, method: str, density: str | None = None, model: str | None = None
) -> list[float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if str(row.get("method")) != method:
            continue
        if density is not None and str(row.get("density")) != density:
            continue
        if model is not None and str(row.get("model")) != model:
            continue
        value = row.get(key)
        if isinstance(value, (int, float)):
            case_key = (str(row.get("model") or ""), str(row.get("density") or ""))
            grouped.setdefault(case_key, []).append(float(value))
    return [float(mean(vals)) for vals in grouped.values() if vals]


def _model_groups(rows: list[dict[str, object]]) -> list[str]:
    present = {str(row.get("model") or "") for row in rows if row.get("model")}
    ordered = [model for model in MODEL_ORDER if model in present]
    return ordered + ["All"]


def _annotate_last_group(
    ax: plt.Axes,
    x_positions: list[float],
    means: list[float],
    errs: list[float],
    *,
    fmt: str,
) -> None:
    ymax = ax.get_ylim()[1]
    lift = ymax * 0.02
    for x, mean_val, err_val in zip(x_positions, means, errs):
        ax.text(
            x,
            mean_val + err_val + lift,
            format(mean_val, fmt),
            ha="center",
            va="bottom",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.25},
            zorder=5,
        )


def _phase1_case_ratios() -> list[dict[str, object]]:
    ratios: list[dict[str, object]] = []
    suite_rows = _read_json(MAIN_RUN / "metadata" / "suite_results.json")
    assert isinstance(suite_rows, list)
    for case in suite_rows:
        if not isinstance(case, dict):
            continue
        if str(case.get("status")) != "ok" or str(case.get("model_key")) == "gemma-2-9b-it":
            continue
        model_key = str(case.get("model_key") or "")
        density = str(case.get("density") or "")
        payload = _read_json(MAIN_RUN / "raw" / density / f"{model_key}_dataset_eval.json")
        assert isinstance(payload, dict)
        base_rows = list((payload.get("per_repeat") or {}).get("phase1_baseline_chunked") or [])
        wave_rows = list((payload.get("per_repeat") or {}).get("phase1") or [])
        case_ratios: list[dict[str, float]] = []
        for base_row, wave_row in zip(base_rows, wave_rows):
            if not isinstance(base_row, dict) or not isinstance(wave_row, dict):
                continue
            base_timings = dict(base_row.get("request_timings") or {})
            wave_hook = dict(wave_row.get("wave_hook_report") or {})
            case_ratios.append(
                {
                    "all_ttft_ratio": _ttft_p99_ms(base_timings) / float(_safe_p99(wave_hook, "ttft_ms_all") or 1.0),
                    "short_ttft_ratio": _ttft_p99_ms(base_timings, is_short=True)
                    / float(_safe_p99(wave_hook, "ttft_ms_short") or 1.0),
                    "round_wall_ratio": float(base_row.get("round_wall_ms") or 0.0)
                    / float(wave_row.get("wave_round_wall_ms") or 1.0),
                }
            )
        if not case_ratios:
            continue
        ratios.append(
            {
                "density": density,
                "model": MODEL_LABELS.get(model_key, model_key),
                "all_ttft_ratio": _mean([item["all_ttft_ratio"] for item in case_ratios]),
                "short_ttft_ratio": _mean([item["short_ttft_ratio"] for item in case_ratios]),
                "round_wall_ratio": _mean([item["round_wall_ratio"] for item in case_ratios]),
            }
        )
    return ratios


def _build_overall_figure(rows: list[dict[str, object]]) -> Path:
    _apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.4), dpi=220, constrained_layout=True)
    metrics = [
        ("all_ttft_p99_ms", "All Requests: p99 TTFT"),
        ("short_ttft_p99_ms", "Short Requests: p99 TTFT"),
        ("round_wall_ms", "Round Completion Time"),
    ]
    groups = _model_groups(rows)
    group_xs = list(range(len(groups)))
    width = 0.18
    offsets = [(-1.5 + idx) * width for idx in range(len(COMPARABLE_METHODS))]
    for ax, (metric, title) in zip(axes, metrics):
        all_group_means: list[float] = []
        all_group_errs: list[float] = []
        ymax = 0.0
        for offset, method in zip(offsets, COMPARABLE_METHODS):
            means = []
            errs = []
            for model in groups:
                scoped_model = None if model == "All" else model
                values = _case_mean_values(rows, key=metric, method=method, model=scoped_model)
                m, ci = _mean_ci95(values)
                means.append(m)
                errs.append(ci)
                ymax = max(ymax, m + ci)
            xs = [x + offset for x in group_xs]
            ax.bar(
                xs,
                means,
                yerr=errs,
                color=METHOD_COLORS[method],
                capsize=3,
                width=width,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
                label=_display_method(method),
            )
            all_group_means.append(means[-1])
            all_group_errs.append(errs[-1])
        ax.set_title(title)
        ax.set_xticks(group_xs, [_display_model(g) for g in groups], rotation=15)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ymax * 1.18)
        ax.axvline(len(groups) - 1.5, color="#A0AEC0", linestyle=":", linewidth=1.0, alpha=0.9)
        if metric.endswith("ms"):
            ax.set_ylabel("Milliseconds")
        else:
            ax.set_ylabel("Ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        framealpha=0.95,
    )
    out_path = CUCUMIS_FIG_DIR / "overall_end_to_end.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_fixed_structure_figure(rows: list[dict[str, object]]) -> Path:
    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=220, constrained_layout=True)
    series = [
        ("all_ttft_p99_ms", "All Requests: p99 TTFT"),
        ("short_ttft_p99_ms", "Short Requests: p99 TTFT"),
    ]
    markers = {"Continuous batching": "o", "Chunked prefill": "s", "Sarathi": "^", "CUCUMIS": "D"}
    xs = list(range(len(DENSITY_ORDER)))
    for ax, (metric, title) in zip(axes, series):
        for method in COMPARABLE_METHODS:
            means = []
            errs = []
            for density in DENSITY_ORDER:
                m, ci = _mean_ci95(_case_mean_values(rows, key=metric, method=method, density=density))
                means.append(m)
                errs.append(ci)
            ax.plot(
                xs,
                means,
                marker=markers[method],
                color=METHOD_COLORS[method],
                linewidth=2.5,
                markersize=8,
                label=_display_method(method),
            )
            ax.fill_between(
                xs,
                [m - e for m, e in zip(means, errs)],
                [m + e for m, e in zip(means, errs)],
                color=METHOD_COLORS[method],
                alpha=0.06,
            )
        ax.set_title(title)
        ax.set_xticks(xs, DENSITY_ORDER)
        ax.set_xlabel("Density")
        ax.set_ylabel("Milliseconds")
        ax.set_axisbelow(True)
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        framealpha=0.95,
        loc="upper left",
    )
    out_path = CUCUMIS_FIG_DIR / "fixed_structure_vs_online_progress.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_mechanism_figure(rows: list[dict[str, object]], phase1_case_ratios: list[dict[str, object]]) -> Path:
    _apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.6), dpi=220, constrained_layout=True)

    groups = _model_groups(rows)
    group_xs = list(range(len(groups)))
    width = 0.18
    offsets = [(-1.5 + idx) * width for idx in range(len(ABLATION_METHODS))]
    for ax, metric, title in (
        (axes[0], "all_ttft_p99_ms", "All Requests: p99 TTFT"),
        (axes[1], "short_ttft_p99_ms", "Short Requests: p99 TTFT"),
    ):
        all_group_means: list[float] = []
        all_group_errs: list[float] = []
        ymax = 0.0
        for offset, method in zip(offsets, ABLATION_METHODS):
            means = []
            errs = []
            for model in groups:
                scoped_model = None if model == "All" else model
                values = _case_mean_values(rows, key=metric, method=method, model=scoped_model)
                m, ci = _mean_ci95(values)
                means.append(m)
                errs.append(ci)
                ymax = max(ymax, m + ci)
            xs = [x + offset for x in group_xs]
            ax.bar(
                xs,
                means,
                yerr=errs,
                color=METHOD_COLORS[method],
                capsize=3,
                width=width,
                error_kw={"elinewidth": 1.2, "capthick": 1.2},
                label=_display_method(method),
            )
            all_group_means.append(means[-1])
            all_group_errs.append(errs[-1])
        ax.set_title(title)
        ax.set_xticks(group_xs, [_display_model(g) for g in groups], rotation=15)
        ax.set_ylabel("Milliseconds")
        ax.set_axisbelow(True)
        ax.set_ylim(0, ymax * 1.18)
        ax.axvline(len(groups) - 1.5, color="#A0AEC0", linestyle=":", linewidth=1.0, alpha=0.9)

    density_means_all = []
    density_errs_all = []
    density_means_short = []
    density_errs_short = []
    for density in DENSITY_ORDER:
        density_rows = [row for row in phase1_case_ratios if str(row.get("density")) == density]
        all_mean, all_ci = _mean_ci95([float(row["all_ttft_ratio"]) for row in density_rows])
        short_mean, short_ci = _mean_ci95([float(row["short_ttft_ratio"]) for row in density_rows])
        density_means_all.append(all_mean)
        density_errs_all.append(all_ci)
        density_means_short.append(short_mean)
        density_errs_short.append(short_ci)
    xs_density = list(range(len(DENSITY_ORDER)))
    axes[2].plot(xs_density, density_means_all, marker="o", linewidth=2.5, markersize=8, color=METHOD_COLORS["CUCUMIS-I"], label="All TTFT ratio")
    axes[2].fill_between(
        xs_density,
        [m - e for m, e in zip(density_means_all, density_errs_all)],
        [m + e for m, e in zip(density_means_all, density_errs_all)],
        color=METHOD_COLORS["CUCUMIS-I"],
        alpha=0.08,
    )
    axes[2].plot(xs_density, density_means_short, marker="s", linewidth=2.5, markersize=8, color=METHOD_COLORS["Chunked prefill"], label="Short TTFT ratio")
    axes[2].fill_between(
        xs_density,
        [m - e for m, e in zip(density_means_short, density_errs_short)],
        [m + e for m, e in zip(density_means_short, density_errs_short)],
        color=METHOD_COLORS["Chunked prefill"],
        alpha=0.08,
    )
    axes[2].axhline(1.0, color="#4A5568", linestyle="--", linewidth=1.0)
    axes[2].set_xticks(xs_density, DENSITY_ORDER)
    axes[2].set_title("Phase-I-Only Local Ratios")
    axes[2].set_ylabel("Improvement ratio")
    axes[2].legend(
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        framealpha=0.95,
        loc="upper right",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        framealpha=0.95,
    )
    out_path = CUCUMIS_FIG_DIR / "mechanism_ablation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _build_long_cost_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for method in LONG_COST_METHODS:
        subset = [row for row in rows if str(row.get("method")) == method]
        out.append(
            {
                "method": _display_method(method),
                "long_ttft_p99_ms": _mean([float(row["long_ttft_p99_ms"]) for row in subset if isinstance(row.get("long_ttft_p99_ms"), (int, float))]),
                "long_slowdown_p99": _mean([float(row["long_slowdown_p99"]) for row in subset if isinstance(row.get("long_slowdown_p99"), (int, float))]),
                "round_wall_ms": _mean([float(row["round_wall_ms"]) for row in subset if isinstance(row.get("round_wall_ms"), (int, float))]),
                "completed_requests_per_s": _mean([float(row["completed_requests_per_s"]) for row in subset if isinstance(row.get("completed_requests_per_s"), (int, float))]),
            }
        )
    return out


def _latex_long_cost(rows: list[dict[str, object]]) -> str:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Long p99 TTFT (ms) & Long p99 slowdown & Total completion time (ms) & Completed req/s \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & {row['long_ttft_p99_ms']:.1f} & {row['long_slowdown_p99']:.2f} & "
            f"{row['round_wall_ms']:.1f} & {row['completed_requests_per_s']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _build_phase1_comparability_table(phase1_case_ratios: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for density in DENSITY_ORDER:
        subset = [row for row in phase1_case_ratios if str(row.get("density")) == density]
        out.append(
            {
                "density": density,
                "all_ttft_ratio": _mean([float(row["all_ttft_ratio"]) for row in subset]),
                "short_ttft_ratio": _mean([float(row["short_ttft_ratio"]) for row in subset]),
                "round_wall_ratio": _mean([float(row["round_wall_ratio"]) for row in subset]),
            }
        )
    return out


def _latex_phase1_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Density & All TTFT & Short TTFT & Wall \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['density']} & {row['all_ttft_ratio']:.2f} & {row['short_ttft_ratio']:.2f} & {row['round_wall_ratio']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _summary_markdown(per_repeat_rows: list[dict[str, object]], long_cost_rows: list[dict[str, object]], phase1_rows: list[dict[str, object]]) -> str:
    def method_line(method: str) -> str:
        subset = [row for row in per_repeat_rows if str(row.get("method")) == method]
        return (
            f"- {_display_method(method)}: all-request p99 TTFT={_mean([float(r['all_ttft_p99_ms']) for r in subset]):.1f} ms, "
            f"short-request p99 TTFT={_mean([float(r['short_ttft_p99_ms']) for r in subset]):.1f} ms, "
            f"all-request p99 slowdown={_mean([float(r['all_slowdown_p99']) for r in subset]):.2f}, "
            f"round wall={_mean([float(r['round_wall_ms']) for r in subset]):.1f} ms"
        )

    lines = [
        "# Chapter 5 Export Summary",
        "",
        f"- Main suite: `{MAIN_RUN / 'metadata' / 'suite_results.json'}`",
        f"- Baseline variants: `{BASELINE_RUN / 'metadata' / 'variant_suite_results.json'}`",
        f"- Exported per-repeat rows: `{len(per_repeat_rows)}`",
        f"- Methods available from current results: `{', '.join(_display_method(m) for m in sorted({str(row['method']) for row in per_repeat_rows}))}`",
        "",
        "## Main takeaways from current export",
        "",
    ]
    for method in ["Continuous batching", "Chunked prefill", "Sarathi", "CUCUMIS-II", "CUCUMIS", "CUCUMIS-Strict"]:
        lines.append(method_line(method))
    lines.extend(
        [
            "",
            "## Phase-I-only local view",
            "",
            f"- Mean Chunked prefill / CUCUMIS-I ratios by density: "
            + ", ".join(
                f"{row['density']} all={row['all_ttft_ratio']:.2f}x short={row['short_ttft_ratio']:.2f}x wall={row['round_wall_ratio']:.2f}x"
                for row in phase1_rows
            ),
            "",
            "## Availability notes",
            "",
            "- Overall, fixed-structure, and mechanism-ablation figures are now regenerated directly from the refreshed seven-model Chapter 5 suite.",
            "- `Sarathi` remains a stack-local substrate-matched serving-policy baseline rather than an official Sarathi-Serve reproduction; this distinction should stay explicit in the paper text.",
            "- `density_sweep.png` and `lora_latency_dispersion.png` are refreshed separately from the dedicated seven-model density and LoRA supporting runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def _manifest_payload(per_repeat_rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "suite_results_json": str(MAIN_RUN / "metadata" / "suite_results.json"),
        "baseline_variants_json": str(BASELINE_RUN / "metadata" / "variant_suite_results.json"),
        "out_dir": str(CUCUMIS_FIG_DIR),
        "exported_per_repeat_rows": len(per_repeat_rows),
        "available_methods": [_display_method(m) for m in sorted({str(row["method"]) for row in per_repeat_rows})],
        "notes": [
            "overall figure centers on the refreshed seven-model end-to-end comparison among vLLM, Chunked prefill, Sarathi, and CUCUMIS.",
            "fixed-structure figure uses the refreshed density-conditioned comparison among vLLM, Chunked prefill, Sarathi, and CUCUMIS.",
            "mechanism ablation combines the comparable CUCUMIS-II / CUCUMIS-Strict view with a separate phase-I-only local-ratio panel for CUCUMIS-I.",
        ],
        "generated_files": [
            "chapter5_manifest.json",
            "chapter5_per_repeat_metrics.csv",
            "chapter5_per_repeat_metrics.json",
            "chapter5_summary.md",
            "density_sweep.png",
            "fixed_structure_vs_online_progress.png",
            "lora_latency_dispersion.png",
            "mechanism_ablation.png",
            "overall_end_to_end.png",
            "table_long_cost_correctness.csv",
            "table_long_cost_correctness.tex",
            "table_phase1_comparability.csv",
            "table_phase1_comparability.tex",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate Chapter 5 main figures, tables, and summary exports.")
    parser.add_argument("--main-run", default=str(DEFAULT_MAIN_RUN))
    parser.add_argument("--baseline-run", default=str(DEFAULT_BASELINE_RUN))
    parser.add_argument("--out-dir", default=str(DEFAULT_CUCUMIS_FIG_DIR))
    args = parser.parse_args()

    global MAIN_RUN, BASELINE_RUN, CUCUMIS_FIG_DIR
    MAIN_RUN = Path(args.main_run).expanduser().resolve()
    BASELINE_RUN = Path(args.baseline_run).expanduser().resolve()
    CUCUMIS_FIG_DIR = Path(args.out_dir).expanduser().resolve()
    if not (MAIN_RUN / "metadata" / "suite_results.json").exists():
        raise FileNotFoundError(f"main suite metadata not found under: {MAIN_RUN}")
    if not (BASELINE_RUN / "metadata" / "variant_suite_results.json").exists():
        raise FileNotFoundError(f"baseline suite metadata not found under: {BASELINE_RUN}")
    CUCUMIS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    per_repeat_rows = _collect_baseline_rows() + _collect_main_rows()
    per_repeat_rows.sort(key=lambda row: (str(row["method"]), str(row["model"]), str(row["density"]), int(row["repeat_index"])))
    phase1_case_ratios = _phase1_case_ratios()
    phase1_case_ratios.sort(key=lambda row: (str(row["density"]), str(row["model"])))

    _build_overall_figure(per_repeat_rows)
    _build_fixed_structure_figure(per_repeat_rows)
    _build_mechanism_figure(per_repeat_rows, phase1_case_ratios)

    per_repeat_json = CUCUMIS_FIG_DIR / "chapter5_per_repeat_metrics.json"
    per_repeat_csv = CUCUMIS_FIG_DIR / "chapter5_per_repeat_metrics.csv"
    _write_json(per_repeat_json, per_repeat_rows)
    _write_csv(per_repeat_csv, per_repeat_rows, list(per_repeat_rows[0].keys()))

    long_cost_rows = _build_long_cost_table(per_repeat_rows)
    _write_csv(CUCUMIS_FIG_DIR / "table_long_cost_correctness.csv", long_cost_rows, list(long_cost_rows[0].keys()))
    (CUCUMIS_FIG_DIR / "table_long_cost_correctness.tex").write_text(_latex_long_cost(long_cost_rows), encoding="utf-8")

    phase1_table_rows = _build_phase1_comparability_table(phase1_case_ratios)
    _write_csv(CUCUMIS_FIG_DIR / "table_phase1_comparability.csv", phase1_table_rows, list(phase1_table_rows[0].keys()))
    (CUCUMIS_FIG_DIR / "table_phase1_comparability.tex").write_text(_latex_phase1_table(phase1_table_rows), encoding="utf-8")

    summary_text = _summary_markdown(per_repeat_rows, long_cost_rows, phase1_table_rows)
    (CUCUMIS_FIG_DIR / "chapter5_summary.md").write_text(summary_text, encoding="utf-8")
    _write_json(CUCUMIS_FIG_DIR / "chapter5_manifest.json", _manifest_payload(per_repeat_rows))

    print(f"updated {CUCUMIS_FIG_DIR / 'overall_end_to_end.png'}")
    print(f"updated {CUCUMIS_FIG_DIR / 'fixed_structure_vs_online_progress.png'}")
    print(f"updated {CUCUMIS_FIG_DIR / 'mechanism_ablation.png'}")
    print(f"updated {per_repeat_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
