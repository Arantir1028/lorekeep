from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt


def aggregate_rows(rows: list[dict[str, Any]], key_fields: list[str], metric_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(field) for field in key_fields)
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, bucket in groups.items():
        agg = {field: value for field, value in zip(key_fields, key)}
        agg["count"] = len(bucket)
        for metric in metric_fields:
            vals = [float(v) for v in (r.get(metric) for r in bucket) if float_or_none(v) is not None]
            if vals:
                agg[f"{metric}_mean"] = sum(vals) / len(vals)
                agg[f"{metric}_min"] = min(vals)
                agg[f"{metric}_max"] = max(vals)
        out.append(agg)
    return out


def plot_tradeoff_scatter(rows: list[dict[str, Any]], out_path: Path) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        return
    plt.figure(figsize=(7, 5))
    for row in ok_rows:
        x = float_or_none(row.get("phase12_wall_improve_mean"))
        y = float_or_none(row.get("phase12_ttft_improve_mean"))
        if x is None or y is None:
            continue
        density = str(row.get("density"))
        label = f"{row.get('model_label')} ({density})"
        plt.scatter(x, y, s=70)
        plt.annotate(label, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Round Wall-Time Improvement (baseline / Wave-Slice)")
    plt.ylabel("TTFT Improvement (baseline / Wave-Slice)")
    plt.title("TTFT vs Wall-Time Tradeoff")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_density_summary(rows: list[dict[str, Any]], out_path: Path) -> None:
    density_rows = aggregate_rows(
        rows,
        key_fields=["density"],
        metric_fields=["phase12_ttft_improve_mean", "phase12_wall_improve_mean"],
    )
    if not density_rows:
        return
    density_rows.sort(key=lambda r: str(r.get("density")))
    names = [str(r["density"]) for r in density_rows]
    ttft = [float(r.get("phase12_ttft_improve_mean_mean") or 0.0) for r in density_rows]
    wall = [float(r.get("phase12_wall_improve_mean_mean") or 0.0) for r in density_rows]
    x = list(range(len(names)))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar([v - width / 2 for v in x], ttft, width=width, label="TTFT")
    plt.bar([v + width / 2 for v in x], wall, width=width, label="Wall")
    plt.xticks(x, names)
    plt.ylabel("Improvement Ratio (baseline / Wave-Slice)")
    plt.title("Mean Tradeoff by Traffic Density")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_rationale_markdown(run_root: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    md: list[str] = []
    md.append("# Open-Workload Execution-Escape Supplement")
    md.append("")
    scenario = config.get("real_world_scenario", {})
    md.append("## Production Scenario")
    md.append("")
    md.append(str(scenario.get("name", "Unnamed scenario")))
    md.append("")
    md.append(str(scenario.get("description", "")))
    md.append("")
    reasons = scenario.get("why_real") or []
    if reasons:
        md.append("Why this scenario is realistic:")
        md.append("")
        for reason in reasons:
            md.append(f"- {reason}")
        md.append("")
    md.append("## Model Selection")
    md.append("")
    for model in config.get("models", []):
        if isinstance(model, dict):
            md.append(f"- {model.get('label', model.get('key'))}: {model.get('reason', '')}")
        else:
            md.append(f"- {model}")
    optional_models = [m for m in config.get("optional_model_extensions", []) if isinstance(m, dict)]
    if optional_models:
        md.append("")
        md.append("Optional model extensions:")
        md.append("")
        for model in optional_models:
            md.append(
                f"- {model.get('label', model.get('key'))}: {model.get('reason', '')} "
                f"(enabled={bool(model.get('enabled', False))})"
            )
    md.append("")
    md.append("## Dataset Selection")
    md.append("")
    for ds in config.get("datasets", []):
        if not isinstance(ds, dict):
            continue
        md.append(f"- {ds.get('label', ds.get('key'))}: {ds.get('role', '')}. {ds.get('reason', '')}")
    optional_datasets = [d for d in config.get("optional_dataset_extensions", []) if isinstance(d, dict)]
    if optional_datasets:
        md.append("")
        md.append("Optional dataset extensions:")
        md.append("")
        for ds in optional_datasets:
            md.append(
                f"- {ds.get('label', ds.get('key'))}: {ds.get('role', '')}. {ds.get('reason', '')} "
                f"(enabled={bool(ds.get('enabled', False))}; note={ds.get('note', '')})"
            )
    md.append("")
    md.append("## Density Design")
    md.append("")
    for density in config.get("workload", {}).get("densities", []):
        if not isinstance(density, dict):
            continue
        md.append(
            f"- {density.get('name')}: phase1={density.get('phase1_arrival_rate')} req/s, "
            f"phase2={density.get('phase2_arrival_rate')} req/s. {density.get('scenario', '')}. {density.get('reason', '')}"
        )
    md.append("")
    md.append("## Evaluation Protocol")
    md.append("")
    eval_cfg = config.get("eval", {})
    md.append(f"- Warmup iterations: {eval_cfg.get('warmup_iters', 'n/a')}")
    md.append(f"- Repeats: {eval_cfg.get('repeats', 'n/a')}")
    md.append("- Arrival process: Poisson")
    md.append("- Mixing policy: beneficiary-rich random short/long release")
    md.append("- Phase-I + Phase-II mode: execution-level escape")
    md.append("")
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        ttft_vals = [float(r["phase12_ttft_improve_mean"]) for r in ok_rows if float_or_none(r.get("phase12_ttft_improve_mean")) is not None]
        wall_vals = [float(r["phase12_wall_improve_mean"]) for r in ok_rows if float_or_none(r.get("phase12_wall_improve_mean")) is not None]
        slow_vals = [float(r["phase12_slowdown_improve_mean"]) for r in ok_rows if float_or_none(r.get("phase12_slowdown_improve_mean")) is not None]
        md.append("## High-Level Result Summary")
        md.append("")
        if ttft_vals:
            md.append(f"- Mean TTFT improvement: {sum(ttft_vals)/len(ttft_vals):.4f}x")
        if wall_vals:
            md.append(f"- Mean round wall-time improvement: {sum(wall_vals)/len(wall_vals):.4f}x")
        if slow_vals:
            md.append(f"- Mean slowdown improvement: {sum(slow_vals)/len(slow_vals):.4f}x")
        md.append(f"- Successful cases: {len(ok_rows)} / {len(rows)}")
        md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- Figures are in English for direct paper use.")
    md.append("- Model and dataset identities are configuration-driven rather than hardcoded in the runner.")
    md.append("- Optional model extensions can be enabled in the JSON config when additional GPU budget is available.")
    (run_root / "metadata" / "experiment_rationale.md").write_text("\n".join(md), encoding="utf-8")


def plot_metric_by_model(rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok" and float_or_none(r.get(metric)) is not None]
    if not ok_rows:
        return
    densities = sorted({str(r.get("density")) for r in ok_rows})
    models = [m for m in dict.fromkeys(str(r.get("model_label")) for r in ok_rows)]
    width = 0.8 / max(1, len(densities))
    x = list(range(len(models)))
    plt.figure(figsize=(max(8, len(models) * 1.5), 5))
    for idx, density in enumerate(densities):
        vals: list[float] = []
        for model in models:
            match = next((r for r in ok_rows if str(r.get("density")) == density and str(r.get("model_label")) == model), None)
            vals.append(float_or_none(match.get(metric)) if match else 0.0)
        xs = [v + idx * width - (len(densities) - 1) * width / 2 for v in x]
        plt.bar(xs, vals, width=width, label=density)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Density")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)


def copy_existing_artifacts(
    rows: list[dict[str, Any]],
    raw_dir: Path,
    repo_root: Path,
    reuse_results_dir: Optional[Path],
    *,
    resolve_existing_result_json: Callable[..., Optional[Path]],
    resolve_existing_meta_json: Callable[[dict[str, Any], Path], Optional[Path]],
    safe_key_fn: Callable[[str], str],
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        result_json = resolve_existing_result_json(row, reuse_results_dir, safe_key_fn=safe_key_fn)
        if result_json is not None:
            dst = raw_dir / result_json.name
            if result_json.resolve() != dst.resolve():
                shutil.copy2(result_json, dst)
            row["result_json"] = str(result_json)
        meta_json = resolve_existing_meta_json(row, repo_root)
        if meta_json is not None:
            dst = raw_dir / meta_json.name
            if meta_json.resolve() != dst.resolve():
                shutil.copy2(meta_json, dst)
            row["workload_meta_json"] = str(meta_json)


def rows_from_existing_csv(csv_path: Path, *, safe_key_fn: Callable[[str], str]) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copied: dict[str, Any] = dict(row)
        for key, value in list(copied.items()):
            if value == "":
                copied[key] = None
        if not copied.get("density"):
            copied["density"] = "mid"
        copied.setdefault("model_key", safe_key_fn(str(copied.get("model") or copied.get("lut_name") or "model")))
        copied.setdefault("model_label", copied.get("model_key"))
        normalized.append(copied)
    return normalized


def enrich_rows_with_config(rows: list[dict[str, Any]], models: list[Any], densities: list[dict[str, Any]]) -> None:
    model_by_id = {m.model_id: m for m in models}
    model_by_key = {m.key: m for m in models}
    density_by_name = {str(d.get("name")): d for d in densities if isinstance(d, dict)}
    for row in rows:
        matched_model = None
        row_model = str(row.get("model") or "")
        row_key = str(row.get("model_key") or "")
        if row_model and row_model in model_by_id:
            matched_model = model_by_id[row_model]
        elif row_key and row_key in model_by_key:
            matched_model = model_by_key[row_key]
        if matched_model is not None:
            row["model_key"] = matched_model.key
            row["model_label"] = matched_model.label
            row["model_reason"] = matched_model.reason
            row["lut_name"] = row.get("lut_name") or matched_model.lut_name
        density_name = str(row.get("density") or "mid")
        density_cfg = density_by_name.get(density_name)
        if density_cfg is not None:
            row["density_scenario"] = density_cfg.get("scenario")
            row["density_reason"] = density_cfg.get("reason")
            row["density_phase1_arrival_rate"] = density_cfg.get("phase1_arrival_rate")
            row["density_phase2_arrival_rate"] = density_cfg.get("phase2_arrival_rate")


def float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None
