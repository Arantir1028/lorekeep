from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from experiments.result_io import read_csv, read_json, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_ROOT = ROOT / "results/chapter5_exports/hardware_portability_a100_4090_5090"
PROFILE_ORDER = ("a100", "rtx4090", "rtx5090")
PROFILE_LABELS = {"a100": "A100", "rtx4090": "RTX 4090", "rtx5090": "RTX 5090"}
PROFILE_ROOTS = {
    profile: ROOT
    / f"results/hardware_portability_remote_mirror/{profile}/results/hardware_portability_supervised_full5/{profile}/unified_distserve_cucumis_2gpu_lb"
    for profile in PROFILE_ORDER
}
RATIOS = (
    ("all_ttft_p99", "distserve_vs_cucumis2_all_ttft_p99_ms_ratio", True, "all p99 TTFT"),
    ("short_ttft_p99", "distserve_vs_cucumis2_short_ttft_p99_ms_ratio", True, "short p99 TTFT"),
    (
        "all_completion_p99",
        "distserve_vs_cucumis2_all_completion_p99_ms_ratio",
        True,
        "all p99 completion",
    ),
    ("round_wall", "distserve_vs_cucumis2_round_wall_ms_ratio", True, "round wall time"),
    ("throughput", "distserve_vs_cucumis2_throughput_rps_ratio", False, "throughput"),
)
METRICS = (
    ("all_ttft_p99_ms", "distserve_all_ttft_p99_ms", "cucumis2_all_ttft_p99_ms"),
    ("short_ttft_p99_ms", "distserve_short_ttft_p99_ms", "cucumis2_short_ttft_p99_ms"),
    ("all_completion_p99_ms", "distserve_all_completion_p99_ms", "cucumis2_all_completion_p99_ms"),
    ("round_wall_ms", "distserve_round_wall_ms", "cucumis2_round_wall_ms"),
    ("throughput_rps", "distserve_throughput_rps", "cucumis2_throughput_rps"),
)
CASE_COLUMNS = [
    "gpu_profile",
    "gpu_label",
    "density",
    "density_level",
    "target_long_fraction_pct",
    "model_key",
    "distserve_decode_replay_mode",
    "distserve_decode_batch_size",
    "distserve_decode_batch_alpha",
    "cucumis_dispatcher",
    "comparison_scope",
    "distserve_all_ttft_p99_ms",
    "cucumis_all_ttft_p99_ms",
    "all_ttft_p99_ratio",
    "distserve_short_ttft_p99_ms",
    "cucumis_short_ttft_p99_ms",
    "short_ttft_p99_ratio",
    "distserve_all_completion_p99_ms",
    "cucumis_all_completion_p99_ms",
    "all_completion_p99_ratio",
    "distserve_round_wall_ms",
    "cucumis_round_wall_ms",
    "round_wall_ratio",
    "distserve_throughput_rps",
    "cucumis_throughput_rps",
    "throughput_ratio",
]


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _number(row: dict[str, str], column: str) -> float:
    if row.get(column, "") == "":
        raise ValueError(f"missing numeric column {column}")
    return float(row[column])


def _wins(value: float, greater: bool) -> bool:
    return value > 1.0 if greater else value < 1.0


def _case_rows(profile: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    direct = (
        "density",
        "density_level",
        "target_long_fraction_pct",
        "model_key",
        "distserve_decode_replay_mode",
        "distserve_decode_batch_size",
        "distserve_decode_batch_alpha",
        "cucumis_dispatcher",
        "comparison_scope",
    )
    value_columns = (
        ("distserve_all_ttft_p99_ms", "distserve_all_ttft_p99_ms"),
        ("cucumis_all_ttft_p99_ms", "cucumis2_all_ttft_p99_ms"),
        ("all_ttft_p99_ratio", RATIOS[0][1]),
        ("distserve_short_ttft_p99_ms", "distserve_short_ttft_p99_ms"),
        ("cucumis_short_ttft_p99_ms", "cucumis2_short_ttft_p99_ms"),
        ("short_ttft_p99_ratio", RATIOS[1][1]),
        ("distserve_all_completion_p99_ms", "distserve_all_completion_p99_ms"),
        ("cucumis_all_completion_p99_ms", "cucumis2_all_completion_p99_ms"),
        ("all_completion_p99_ratio", RATIOS[2][1]),
        ("distserve_round_wall_ms", "distserve_round_wall_ms"),
        ("cucumis_round_wall_ms", "cucumis2_round_wall_ms"),
        ("round_wall_ratio", RATIOS[3][1]),
        ("distserve_throughput_rps", "distserve_throughput_rps"),
        ("cucumis_throughput_rps", "cucumis2_throughput_rps"),
        ("throughput_ratio", RATIOS[4][1]),
    )
    output = []
    for row in rows:
        normalized = {
            "gpu_profile": profile,
            "gpu_label": row.get("hardware_label") or PROFILE_LABELS[profile],
        }
        normalized.update((key, row[key]) for key in direct)
        normalized.update((target, _fmt(_number(row, source))) for target, source in value_columns)
        output.append(normalized)
    return output


def _summary(profile: str, rows: list[dict[str, str]], metadata: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "gpu_profile": profile,
        "gpu_label": metadata.get("hardware_label") or PROFILE_LABELS[profile],
        "cases": len(rows),
        "models": len({row["model_key"] for row in rows}),
        "workloads": len({row["density"] for row in rows}),
        "distserve_decode_replay_mode": ",".join(
            sorted({row["distserve_decode_replay_mode"] for row in rows})
        ),
        "cucumis_dispatcher": ",".join(sorted({row["cucumis_dispatcher"] for row in rows})),
        "gpu_count": metadata.get("gpu_count", ""),
        "physical_measurement_gpu_count": "1",
        "host_label": metadata.get("host_label", ""),
        "gpu_memory_utilization_override": metadata.get("gpu_memory_utilization_override", ""),
        "gpu_memory_utilization_by_model": metadata.get("gpu_memory_utilization_by_model", ""),
        "max_num_batched_tokens_by_model": metadata.get("max_num_batched_tokens_by_model", ""),
        "elapsed_sec": _fmt(float(metadata.get("elapsed_sec", 0.0)))
        if metadata.get("elapsed_sec")
        else "",
    }
    for key, column, greater, _ in RATIOS:
        values = [_number(row, column) for row in rows]
        output.update(
            {
                f"{key}_ratio_mean": _fmt(mean(values)),
                f"{key}_ratio_median": _fmt(median(values)),
                f"{key}_win_cases": sum(_wins(value, greater) for value in values),
                f"{key}_win_total": len(values),
            }
        )
    for metric, distserve, cucumis in METRICS:
        output[f"distserve_{metric}_mean"] = _fmt(mean(_number(row, distserve) for row in rows))
        output[f"cucumis_{metric}_mean"] = _fmt(mean(_number(row, cucumis) for row in rows))
    return output


def _trends(profile: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["target_long_fraction_pct"]].append(row)
    output = []
    for long_pct in sorted(groups, key=int):
        group = groups[long_pct]
        item = {
            "gpu_profile": profile,
            "gpu_label": PROFILE_LABELS[profile],
            "target_long_fraction_pct": long_pct,
            "cases": len(group),
        }
        item.update(
            (f"{key}_ratio_mean", _fmt(mean(_number(row, column) for row in group)))
            for key, column, _, _ in RATIOS
        )
        output.append(item)
    return output


def _note(profile: str, losing: list[str]) -> str:
    if profile == "rtx4090":
        return "RTX 4090 is the tightest-memory profile; treat mixed TTFT as a stress case and avoid universal-win wording."
    if profile == "rtx5090" and "short p99 TTFT" in losing:
        return "RTX 5090 short-TTFT exception; completion, wall time, and throughput still win in the completed run."
    if profile == "rtx5090":
        return "RTX 5090 all-TTFT mixed case; short-TTFT and guardrail metrics usually still win."
    return "A100 all-TTFT exception; short-TTFT and guardrail metrics stay winning in this run."


def _anomalies(profile: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        losing = [
            label
            for _, column, greater, label in RATIOS
            if not _wins(_number(row, column), greater)
        ]
        if not losing:
            continue
        output.append(
            {
                "gpu_profile": profile,
                "gpu_label": PROFILE_LABELS[profile],
                "model_key": row["model_key"],
                "density": row["density"],
                "density_level": row["density_level"],
                "target_long_fraction_pct": row["target_long_fraction_pct"],
                "losing_metrics": "; ".join(losing),
                "severity": "spot_rerun_candidate"
                if "short p99 TTFT" in losing
                or any(
                    label in losing
                    for label in ("all p99 completion", "round wall time", "throughput")
                )
                else "paper_caveat",
                "all_ttft_p99_ratio": _fmt(_number(row, RATIOS[0][1])),
                "short_ttft_p99_ratio": _fmt(_number(row, RATIOS[1][1])),
                "all_completion_p99_ratio": _fmt(_number(row, RATIOS[2][1])),
                "round_wall_ratio": _fmt(_number(row, RATIOS[3][1])),
                "throughput_ratio": _fmt(_number(row, RATIOS[4][1])),
                "note": _note(profile, losing),
            }
        )
    return sorted(
        output,
        key=lambda item: (
            item["gpu_profile"],
            item["severity"] != "spot_rerun_candidate",
            float(item["all_ttft_p99_ratio"]),
            item["model_key"],
            item["density"],
        ),
    )


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hardware portability under equal-resource DistServe/CUCUMIS comparison. Ratios are DistServe/CUCUMIS; for latency and wall time, higher is better for CUCUMIS, while throughput below $1\times$ means CUCUMIS is faster.}",
        r"\label{tab:hardware-portability}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"GPU & All TTFT p99 & Short TTFT p99 & Completion p99 & Wall time & Throughput \\",
        r"\midrule",
    ]
    for row in rows:
        values = [row["gpu_label"]] + [
            f"{float(row[f'{key}_ratio_mean']):.2f}x" for key, *_ in RATIOS
        ]
        lines.append(" & ".join(values) + r" \\")
    path.write_text(
        "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]), encoding="utf-8"
    )


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| GPU | cases | all TTFT p99 | short TTFT p99 | completion p99 | wall | throughput | TTFT wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ratios = [f"{float(row[f'{key}_ratio_mean']):.2f}x" for key, *_ in RATIOS]
        lines.append(
            f"| {row['gpu_label']} | {row['cases']} | {' | '.join(ratios)} | {row['all_ttft_p99_win_cases']} / {row['all_ttft_p99_win_total']} |"
        )
    return "\n".join(lines)


def _write_readme(
    path: Path, summaries: list[dict[str, Any]], anomalies: list[dict[str, Any]]
) -> None:
    counts, short_counts = defaultdict(int), defaultdict(int)
    for row in anomalies:
        counts[row["gpu_label"]] += 1
        short_counts[row["gpu_label"]] += "short p99 TTFT" in row["losing_metrics"]
    lines = [
        "# Frozen Result: Hardware Portability A100 / RTX 4090 / RTX 5090",
        "",
        "This directory is the paper-facing export root for `hardware_portability_a100_4090_5090`.",
        "",
        "## Status",
        "",
        "- Complete: yes",
        "- Main cases: 3 GPU profiles x 50 cases",
        "- Methods: `DistServe-2GPU` and `CUCUMIS-2GPU-LB`",
        "- DistServe semantics: single-GPU physical stage-cost measurement, token-level continuous-batching replay, equal-resource logical 2-GPU comparison",
        "- CUCUMIS semantics: two real CUCUMIS replicas on the same two local GPUs, least-backlog dispatch",
        "",
        "## Important Files",
        "",
        "- `hardware_portability_summary.csv`: compact aggregate table",
        "- `hardware_portability_summary.tex`: LaTeX table for paper drafting",
        "- `hardware_portability_case_comparison.csv`: normalized 150-case comparison table",
        "- `hardware_portability_long_ratio_trend.csv`: ratio trend by long-request fraction",
        "- `hardware_portability_anomalies.csv`: cases where at least one paper-facing metric does not favor CUCUMIS",
        "- `hardware_portability_manifest.json`: source-result provenance",
        "",
        "## Aggregate Results",
        "",
        _markdown(summaries),
        "",
        "Ratios are `DistServe / CUCUMIS`. For latency and wall time, larger than 1 means CUCUMIS is better. For throughput, smaller than 1 means CUCUMIS has higher throughput.",
        "",
        "## Caveats",
        "",
        "- Do not write that CUCUMIS wins every workload. Use average and win-count wording.",
        "- RTX 4090 is the memory-stress profile; its all-request TTFT is mixed, while completion, wall time, and throughput still favor CUCUMIS on most cases.",
        "- RTX 5090 has only a few short-TTFT exceptions, concentrated in Gemma-7B cases; completion, wall time, and throughput win in all 50 cases.",
        "",
        "Exception counts by GPU:",
        "",
    ]
    lines.extend(
        f"- {row['gpu_label']}: {counts[row['gpu_label']]} cases with at least one losing metric; {short_counts[row['gpu_label']]} short-TTFT exceptions."
        for row in summaries
    )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "This is a historical result bundle. The full source run trees are not tracked,",
            "and the measurements must not be presented as results from the current",
            "WaveSlice source revision without a new full sweep.",
            "",
            "See [the repository result notes](../../../docs/results.md) for provenance and",
            "data-availability rules.",
        ]
    )
    path.write_text("\n".join(lines + [""]), encoding="utf-8")


def summarize(export_root: Path) -> None:
    export_root.mkdir(parents=True, exist_ok=True)
    export_label = (
        str(export_root.relative_to(ROOT)) if export_root.is_relative_to(ROOT) else str(export_root)
    )
    cases: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    trends: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    sources = {}
    for profile in PROFILE_ORDER:
        root = PROFILE_ROOTS[profile]
        comparison, metadata_path = (
            root / "cucumis_2a100/distserve_equal_resource_real_comparison.csv",
            root / "metadata/hardware_profile.json",
        )
        rows = read_csv(comparison)
        if len(rows) != 50:
            raise ValueError(f"{profile} expected 50 comparison rows, found {len(rows)}")
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        cases.extend(_case_rows(profile, rows))
        summaries.append(_summary(profile, rows, metadata))
        trends.extend(_trends(profile, rows))
        anomalies.extend(_anomalies(profile, rows))
        sources[profile] = {
            "root": str(root.relative_to(ROOT)),
            "comparison_csv": str(comparison.relative_to(ROOT)),
            "metadata_json": str(metadata_path.relative_to(ROOT)),
            "status": metadata.get("status", ""),
            "host_label": metadata.get("host_label", ""),
            "elapsed_sec": metadata.get("elapsed_sec", ""),
        }
    summary_columns = [
        "gpu_profile",
        "gpu_label",
        "cases",
        "models",
        "workloads",
        "distserve_decode_replay_mode",
        "cucumis_dispatcher",
        "gpu_count",
        "physical_measurement_gpu_count",
        "host_label",
        "gpu_memory_utilization_override",
        "gpu_memory_utilization_by_model",
        "max_num_batched_tokens_by_model",
        "elapsed_sec",
    ]
    for key, *_ in RATIOS:
        summary_columns.extend(
            f"{key}_{suffix}" for suffix in ("ratio_mean", "ratio_median", "win_cases", "win_total")
        )
    summary_columns.extend(
        f"{system}_{metric}_mean" for metric, *_ in METRICS for system in ("distserve", "cucumis")
    )
    trend_columns = ["gpu_profile", "gpu_label", "target_long_fraction_pct", "cases"] + [
        f"{key}_ratio_mean" for key, *_ in RATIOS
    ]
    anomaly_columns = [
        "gpu_profile",
        "gpu_label",
        "model_key",
        "density",
        "density_level",
        "target_long_fraction_pct",
        "losing_metrics",
        "severity",
        "all_ttft_p99_ratio",
        "short_ttft_p99_ratio",
        "all_completion_p99_ratio",
        "round_wall_ratio",
        "throughput_ratio",
        "note",
    ]
    write_csv(export_root / "hardware_portability_summary.csv", summaries, summary_columns)
    write_csv(export_root / "hardware_portability_case_comparison.csv", cases, CASE_COLUMNS)
    write_csv(export_root / "hardware_portability_long_ratio_trend.csv", trends, trend_columns)
    write_csv(export_root / "hardware_portability_anomalies.csv", anomalies, anomaly_columns)
    _write_latex(export_root / "hardware_portability_summary.tex", summaries)
    _write_readme(export_root / "README.md", summaries, anomalies)
    write_json(
        export_root / "hardware_portability_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "export_root": export_label,
            "profiles": list(PROFILE_ORDER),
            "sources": sources,
            "ratio_definition": "DistServe / CUCUMIS",
            "latency_ratio_interpretation": "larger than 1 favors CUCUMIS",
            "throughput_ratio_interpretation": "smaller than 1 favors CUCUMIS",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    export_root = parser.parse_args().export_root
    summarize(export_root if export_root.is_absolute() else ROOT / export_root)
    print(f"Wrote hardware portability exports to {export_root}")


if __name__ == "__main__":
    main()
