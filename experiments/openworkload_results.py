from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.result_io import safe_float as float_or_none


def aggregate_rows(
    rows: list[dict[str, Any]], key_fields: list[str], metric_fields: list[str]
) -> list[dict[str, Any]]:
    groups = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault(tuple(row.get(field) for field in key_fields), []).append(row)
    output = []
    for key, bucket in groups.items():
        aggregate = {field: value for field, value in zip(key_fields, key, strict=False)} | {
            "count": len(bucket)
        }
        for metric in metric_fields:
            values = [
                float(value)
                for value in (row.get(metric) for row in bucket)
                if float_or_none(value) is not None
            ]
            if values:
                aggregate.update(
                    {
                        f"{metric}_mean": sum(values) / len(values),
                        f"{metric}_min": min(values),
                        f"{metric}_max": max(values),
                    }
                )
        output.append(aggregate)
    return output


def _values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [float(row[metric]) for row in rows if float_or_none(row.get(metric)) is not None]


def write_result_summary_markdown(run_root: Path, rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# V1 Open-Workload Result Summary",
        "",
        f"- Successful cases: {len(successful)} / {len(rows)}",
        "",
    ]
    if not successful:
        lines.append("No successful rows were recorded.")
    else:
        for label, prefix in (
            ("Phase I", "phase1"),
            ("Phase II", "phase2"),
            ("Phase I+II", "phase12"),
        ):
            ttft, wall = (
                _values(successful, f"{prefix}_ttft_improve_mean"),
                _values(successful, f"{prefix}_wall_improve_mean"),
            )
            if not ttft and not wall:
                continue
            lines += [f"## {label}", ""]
            for name, values in (("TTFT", ttft), ("wall-time", wall)):
                if values:
                    lines += [
                        f"- Mean {name} improvement: {sum(values) / len(values):.4f}x",
                        f"- Min {name} improvement: {min(values):.4f}x",
                        f"- Max {name} improvement: {max(values):.4f}x",
                    ]
            lines.append("")
        lines += ["## Best Phase I+II Cases", ""]
        for row in sorted(
            successful,
            key=lambda item: float_or_none(item.get("phase12_ttft_improve_mean")) or 0,
            reverse=True,
        )[:5]:
            lines.append(
                f"- {row.get('model_label')} / {row.get('density')}: Phase I+II TTFT={float_or_none(row.get('phase12_ttft_improve_mean')) or 0:.4f}x, wall={float_or_none(row.get('phase12_wall_improve_mean')) or 0:.4f}x, Phase I TTFT={float_or_none(row.get('phase1_ttft_improve_mean')) or 0:.4f}x, Phase II TTFT={float_or_none(row.get('phase2_ttft_improve_mean')) or 0:.4f}x"
            )
        lines.append("")
    (run_root / "metadata/result_summary.md").write_text("\n".join(lines), encoding="utf-8")
