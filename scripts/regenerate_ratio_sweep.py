from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt

from experiments.result_io import read_json, safe_float, timing_summary, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
RATIO_DENSITY_RE = re.compile(r"^(?P<level>[a-z]+)_l(?P<long_pct>\d{1,2})$")
METHOD_COLORS = {
    "vLLM": "#4C78A8",
    "Chunked prefill": "#F58518",
    "Sarathi": "#72B7B2",
    "CUCUMIS": "#E45756",
    "Best chunk-aware": "#6B7280",
}


def _resolve_path(value: Any, *, run_root: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    repo_path = ROOT / path
    return repo_path if repo_path.exists() else run_root / path


def _ratio(num: float | None, den: float | None) -> float | None:
    return None if num is None or den is None or den <= 0 else float(num) / float(den)


def _mean(values: list[float | None]) -> float | None:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(data) / len(data)) if data else None


def _mean_ci95(values: list[float | None]) -> tuple[float | None, float | None]:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not data:
        return None, None
    return (
        (data[0], 0.0)
        if len(data) == 1
        else (float(mean(data)), float(1.96 * stdev(data) / math.sqrt(len(data))))
    )


def _density_info(name: str) -> dict[str, object] | None:
    match = RATIO_DENSITY_RE.match(name)
    return (
        None
        if not match
        else {"density_level": match["level"], "target_long_fraction_pct": int(match["long_pct"])}
    )


def _repeat_metric_row(
    *,
    density: str,
    model_key: str,
    model_label: str,
    method: str,
    repeat_index: int,
    request_timings: Any,
    round_wall_ms: Any,
    source_json: Path,
) -> dict[str, object]:
    timing, wall = (
        timing_summary(request_timings, include_fraction=True, float_counts=True),
        safe_float(round_wall_ms),
    )
    count, info = safe_float(timing.get("request_count")), _density_info(density) or {}
    row: dict[str, object] = {
        "density": density,
        "density_level": info.get("density_level", ""),
        "target_long_fraction_pct": info.get("target_long_fraction_pct", ""),
        "model_key": model_key,
        "model_label": model_label,
        "method": method,
        "repeat_index": repeat_index,
        "round_wall_ms": wall,
        "throughput_rps": count * 1000.0 / wall
        if count is not None and wall and wall > 0
        else None,
        "source_json": str(source_json),
    }
    row.update(timing)
    return row


def _append_repeats(
    output: list[dict[str, object]],
    *,
    repeats: list[Any],
    density: str,
    model_key: str,
    model_label: str,
    method: str,
    timing_key: str,
    wall_key: str,
    source: Path,
) -> None:
    for index, item in enumerate(repeats):
        if isinstance(item, dict):
            output.append(
                _repeat_metric_row(
                    density=density,
                    model_key=model_key,
                    model_label=model_label,
                    method=method,
                    repeat_index=index,
                    request_timings=item.get(timing_key),
                    round_wall_ms=item.get(wall_key),
                    source_json=source,
                )
            )


def _suite_payloads(
    run: Path, metadata_name: str
) -> list[tuple[dict[str, Any], Path, dict[str, Any]]]:
    metadata = run / "metadata" / metadata_name
    rows = read_json(metadata)
    if not isinstance(rows, list):
        raise ValueError(f"expected list in {metadata}")
    output = []
    for row in rows:
        density = str(row.get("density") or "") if isinstance(row, dict) else ""
        if (
            not isinstance(row, dict)
            or str(row.get("status")) != "ok"
            or _density_info(density) is None
        ):
            continue
        path = _resolve_path(row.get("result_json"), run_root=run)
        if path.exists() and isinstance((payload := read_json(path)), dict):
            output.append((row, path, payload))
    return output


def _load_main_rows(main_run: Path) -> list[dict[str, object]]:
    output = []
    for row, path, payload in _suite_payloads(main_run, "suite_results.json"):
        _append_repeats(
            output,
            repeats=list((payload.get("per_repeat") or {}).get("phase12") or []),
            density=str(row.get("density") or ""),
            model_key=str(row.get("model_key") or ""),
            model_label=str(row.get("model_label") or row.get("model_key") or ""),
            method="CUCUMIS",
            timing_key="wave_request_timings",
            wall_key="wave_round_wall_ms",
            source=path,
        )
    return output


def _load_baseline_rows(baseline_run: Path) -> list[dict[str, object]]:
    output = []
    specs = {
        "fixed_chunk_vs_sarathi": (
            ("phase2", "Chunked prefill", "base"),
            ("phase2", "Sarathi", "wave"),
        ),
        "strict_no_chunk": (
            ("phase2", "vLLM", "base"),
            ("phase2", "CUCUMIS-II", "wave"),
            ("phase2_strict", "CUCUMIS-Strict", "strict"),
        ),
    }
    for row, path, payload in _suite_payloads(baseline_run, "variant_suite_results.json"):
        per_repeat = dict(payload.get("per_repeat") or {})
        common = {
            "density": str(row.get("density") or ""),
            "model_key": str(row.get("model_key") or ""),
            "model_label": str(row.get("model_label") or row.get("model_key") or ""),
            "source": path,
        }
        variant_specs = specs.get(str(row.get("variant_key") or ""), ())
        if variant_specs and variant_specs[0][0] == "phase2":
            phase2_specs = [spec for spec in variant_specs if spec[0] == "phase2"]
            repeats = list(per_repeat.get("phase2") or [])
            for index, item in enumerate(repeats):
                if not isinstance(item, dict):
                    continue
                for _, method, prefix in phase2_specs:
                    output.append(
                        _repeat_metric_row(
                            **{key: value for key, value in common.items() if key != "source"},
                            method=method,
                            repeat_index=index,
                            request_timings=item.get(f"{prefix}_request_timings"),
                            round_wall_ms=item.get(f"{prefix}_round_wall_ms"),
                            source_json=path,
                        )
                    )
            for repeat_key, method, prefix in variant_specs:
                if repeat_key != "phase2":
                    _append_repeats(
                        output,
                        repeats=list(per_repeat.get(repeat_key) or []),
                        method=method,
                        timing_key=f"{prefix}_request_timings",
                        wall_key=f"{prefix}_round_wall_ms",
                        **common,
                    )
    return output


_METHOD_METRICS = (
    "measured_long_fraction",
    "request_count",
    "short_request_count",
    "long_request_count",
    "all_ttft_p50_ms",
    "all_ttft_p90_ms",
    "all_ttft_p99_ms",
    "short_ttft_p50_ms",
    "short_ttft_p90_ms",
    "short_ttft_p99_ms",
    "long_ttft_p50_ms",
    "long_ttft_p90_ms",
    "long_ttft_p99_ms",
    "all_completion_p50_ms",
    "all_completion_p90_ms",
    "all_completion_p99_ms",
    "short_completion_p50_ms",
    "short_completion_p90_ms",
    "short_completion_p99_ms",
    "long_completion_p50_ms",
    "long_completion_p90_ms",
    "long_completion_p99_ms",
    "round_wall_ms",
    "throughput_rps",
)


def _aggregate_method_rows(repeat_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in repeat_rows:
        groups[str(row.get("density")), str(row.get("model_key")), str(row.get("method"))].append(
            row
        )
    output = []
    for (density, model_key, method), items in sorted(groups.items()):
        first = items[0]
        row: dict[str, object] = {
            "density": density,
            "density_level": first.get("density_level", ""),
            "target_long_fraction_pct": first.get("target_long_fraction_pct", ""),
            "model_key": model_key,
            "model_label": first.get("model_label", model_key),
            "method": method,
            "repeat_count": len(items),
        }
        row.update(
            {key: _mean([safe_float(item.get(key)) for item in items]) for key in _METHOD_METRICS}
        )
        output.append(row)
    return output


def _comparison_rows(method_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    cases: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in method_rows:
        cases[str(row.get("density")), str(row.get("model_key"))][str(row.get("method"))] = row
    output = []
    for (density, model_key), methods in sorted(cases.items()):
        cucumis, vllm = methods.get("CUCUMIS"), methods.get("vLLM")
        if not cucumis or not vllm:
            continue
        chunks = [methods[name] for name in ("Chunked prefill", "Sarathi") if name in methods]
        row: dict[str, object] = {
            "density": density,
            "density_level": cucumis.get("density_level", ""),
            "target_long_fraction_pct": cucumis.get("target_long_fraction_pct", ""),
            "measured_long_fraction": cucumis.get("measured_long_fraction"),
            "model_key": model_key,
            "model_label": cucumis.get("model_label", model_key),
        }
        for metric in (
            "all_ttft_p99_ms",
            "short_ttft_p99_ms",
            "all_completion_p99_ms",
            "short_completion_p99_ms",
        ):
            cuc, vl = safe_float(cucumis.get(metric)), safe_float(vllm.get(metric))
            candidates = [
                value for item in chunks if (value := safe_float(item.get(metric))) is not None
            ]
            best = min(candidates) if candidates else None
            row.update(
                {
                    f"cucumis_{metric}": cuc,
                    f"vllm_{metric}": vl,
                    f"best_chunk_{metric}": best,
                    f"cucumis_vs_vllm_{metric}_improvement": _ratio(vl, cuc),
                    f"cucumis_vs_best_chunk_{metric}_improvement": _ratio(best, cuc),
                }
            )
        cuc_wall, vllm_wall = (
            safe_float(cucumis.get("round_wall_ms")),
            safe_float(vllm.get("round_wall_ms")),
        )
        cuc_throughput, vllm_throughput = (
            safe_float(cucumis.get("throughput_rps")),
            safe_float(vllm.get("throughput_rps")),
        )
        row.update(
            cucumis_round_wall_ms=cuc_wall,
            vllm_round_wall_ms=vllm_wall,
            cucumis_vs_vllm_round_wall_ratio=_ratio(cuc_wall, vllm_wall),
            cucumis_throughput_rps=cuc_throughput,
            vllm_throughput_rps=vllm_throughput,
            cucumis_vs_vllm_throughput_ratio=_ratio(cuc_throughput, vllm_throughput),
        )
        output.append(row)
    return output


_SUMMARY_METRICS = (
    "measured_long_fraction",
    "cucumis_vs_vllm_all_ttft_p99_ms_improvement",
    "cucumis_vs_vllm_short_ttft_p99_ms_improvement",
    "cucumis_vs_best_chunk_all_ttft_p99_ms_improvement",
    "cucumis_vs_best_chunk_short_ttft_p99_ms_improvement",
    "cucumis_vs_vllm_all_completion_p99_ms_improvement",
    "cucumis_vs_vllm_short_completion_p99_ms_improvement",
    "cucumis_vs_vllm_round_wall_ratio",
    "cucumis_vs_vllm_throughput_ratio",
)


def _summary_rows(comparison_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in comparison_rows:
        key = str(row.get("density_level") or ""), int(row.get("target_long_fraction_pct") or 0)
        if all(key):
            groups[key].append(row)
    output = []
    for (level, pct), items in sorted(groups.items()):
        row: dict[str, object] = {
            "density_level": level,
            "target_long_fraction_pct": pct,
            "model_count": len({str(item.get("model_key")) for item in items}),
            "case_count": len(items),
        }
        for key in _SUMMARY_METRICS:
            average, ci = _mean_ci95([safe_float(item.get(key)) for item in items])
            row[f"{key}_mean"], row[f"{key}_ci95"] = average, ci
        output.append(row)
    return output


def _guardrail_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "density_level",
        "target_long_fraction_pct",
        "model_count",
        "case_count",
        "measured_long_fraction_mean",
        "cucumis_vs_vllm_round_wall_ratio_mean",
        "cucumis_vs_vllm_round_wall_ratio_ci95",
        "cucumis_vs_vllm_throughput_ratio_mean",
        "cucumis_vs_vllm_throughput_ratio_ci95",
        "cucumis_vs_vllm_all_completion_p99_ms_improvement_mean",
        "cucumis_vs_vllm_short_completion_p99_ms_improvement_mean",
    )
    return [{key: row.get(key) for key in fields} for row in summary_rows]


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )


def _plot(summary: list[dict[str, object]], out_path: Path, *, guardrail: bool) -> None:
    _apply_style()
    levels = [
        level
        for level in ("mid", "high", "peak")
        if any(row.get("density_level") == level for row in summary)
    ]
    if not levels:
        return
    figure, axes = plt.subplots(
        1,
        len(levels),
        figsize=(6.2 * len(levels), 4.4 if guardrail else 4.8),
        dpi=220,
        squeeze=False,
    )
    series = (
        (
            ("cucumis_vs_vllm_round_wall_ratio", "Round time ratio", "#4C78A8", "o"),
            ("cucumis_vs_vllm_throughput_ratio", "Throughput ratio", "#54A24B", "s"),
        )
        if guardrail
        else (
            (
                "cucumis_vs_vllm_all_ttft_p99_ms_improvement",
                "All vs vLLM",
                METHOD_COLORS["vLLM"],
                "o",
            ),
            (
                "cucumis_vs_vllm_short_ttft_p99_ms_improvement",
                "Short vs vLLM",
                METHOD_COLORS["CUCUMIS"],
                "s",
            ),
            (
                "cucumis_vs_best_chunk_short_ttft_p99_ms_improvement",
                "Short vs best chunk-aware",
                METHOD_COLORS["Best chunk-aware"],
                "^",
            ),
        )
    )
    for axis, level in zip(axes[0], levels, strict=False):
        rows = sorted(
            (row for row in summary if row.get("density_level") == level),
            key=lambda row: int(row.get("target_long_fraction_pct") or 0),
        )
        xs = [100 * float(row.get("measured_long_fraction_mean") or 0) for row in rows]
        for key, label, color, marker in series:
            ys = [safe_float(row.get(f"{key}_mean")) for row in rows]
            cis = [safe_float(row.get(f"{key}_ci95")) or 0 for row in rows]
            axis.plot(xs, ys, marker=marker, color=color, linewidth=2.2, markersize=6, label=label)
            axis.fill_between(
                xs,
                [(y or 0) - ci for y, ci in zip(ys, cis, strict=False)],
                [(y or 0) + ci for y, ci in zip(ys, cis, strict=False)],
                color=color,
                alpha=0.08,
            )
        axis.axhline(1, color="#4A5568", linestyle="--", linewidth=1)
        axis.set_title(f"{level.capitalize()} density")
        axis.set_xlabel("Measured long-request fraction (%)")
        axis.set_ylabel("CUCUMIS / vLLM ratio" if guardrail else "p99 TTFT improvement ratio")
        axis.set_xticks([10, 30, 50, 70, 90])
        axis.set_xlim(7, 93)
        axis.set_axisbelow(True)
    axes[0][0].legend(frameon=True, facecolor="white", edgecolor="#D9D9D9", framealpha=0.95)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)


def _plot_ttft(summary_rows: list[dict[str, object]], out_path: Path) -> None:
    _plot(summary_rows, out_path, guardrail=False)


def _plot_guardrail(summary_rows: list[dict[str, object]], out_path: Path) -> None:
    _plot(summary_rows, out_path, guardrail=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate Chapter 5 request-ratio sweep summaries and figures."
    )
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--out-dir", default="results/chapter5_exports/ratio_sweep")
    args = parser.parse_args()
    main_run, baseline_run, out_dir = (
        _resolve_path(value, run_root=ROOT).resolve()
        for value in (args.main_run, args.baseline_run, args.out_dir)
    )
    if not main_run.exists() or not baseline_run.exists():
        raise FileNotFoundError(
            f"{'main' if not main_run.exists() else 'baseline'} run not found: {main_run if not main_run.exists() else baseline_run}"
        )
    repeat_rows = _load_baseline_rows(baseline_run) + _load_main_rows(main_run)
    if not repeat_rows:
        raise RuntimeError("no ratio-sweep rows found")
    method_rows = _aggregate_method_rows(repeat_rows)
    comparison, summary = _comparison_rows(method_rows), None
    summary = _summary_rows(comparison)
    outputs = {
        "ratio_sweep_repeat_metrics": repeat_rows,
        "ratio_sweep_method_metrics": method_rows,
        "ratio_sweep_per_model_comparison": comparison,
        "ratio_sweep_summary": summary,
    }
    for name, rows in outputs.items():
        write_json(out_dir / f"{name}.json", rows)
        write_csv(out_dir / f"{name}.csv", rows)
    write_csv(out_dir / "ratio_sweep_guardrails.csv", _guardrail_rows(summary))
    _plot_ttft(summary, out_dir / "ratio_sweep_ttft_improvement.pdf")
    _plot_guardrail(summary, out_dir / "ratio_sweep_guardrails.pdf")
    print(f"[RatioSweep] wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
