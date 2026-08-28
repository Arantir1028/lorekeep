from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.result_io import (
    safe_float as float_or_none,
    write_csv as write_csv,
    write_json as write_json,
)

__all__ = ("float_or_none", "write_csv", "write_json")
REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return REPO_ROOT


def project_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base or REPO_ROOT) / path


def load_config(path: str) -> dict[str, Any]:
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def relative_to_repo(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return (
        str(resolved.relative_to(REPO_ROOT))
        if resolved.is_relative_to(REPO_ROOT)
        else str(Path(path))
    )


def resource_policy(config: dict[str, Any]) -> dict[str, Any]:
    selection, resources = config.get("resource_selection") or {}, config.get("resources") or {}
    return {
        "auto_download": bool(selection.get("auto_download", resources.get("auto_download", True))),
        "offline": bool(selection.get("offline", resources.get("offline", False))),
    }


def apply_hf_resource_env(env: dict[str, str], config: dict[str, Any]) -> dict[str, str]:
    policy, names = (
        resource_policy(config),
        ("HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"),
    )
    if policy["offline"]:
        env.update((name, "1") for name in names)
    elif policy["auto_download"]:
        for name in names:
            env.pop(name, None)
    return env


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def workload_meta_matches_model(
    *,
    meta_path: Path,
    model: Any,
    model_path: str,
    local_snapshot: str | None,
    density: dict[str, Any],
    workload_cfg: dict[str, Any],
    require_density_match: bool = True,
) -> bool:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    allowed = {str(model.model_id), str(model_path)} | (
        {str(local_snapshot)} if local_snapshot else set()
    )
    expected = (
        (str(meta.get("model_path")), allowed),
        (bool(meta.get("trust_remote_code", False)), {bool(model.trust_remote_code)}),
        (str(meta.get("arrival_mode")), {str(workload_cfg.get("arrival_mode", "poisson"))}),
        (
            str(meta.get("phase1_arrival_layout")),
            {str(workload_cfg.get("phase1_arrival_layout", "beneficiary_rich"))},
        ),
        (
            str(meta.get("phase2_arrival_layout")),
            {str(workload_cfg.get("phase2_arrival_layout", "beneficiary_rich"))},
        ),
    )
    if any(value not in choices for value, choices in expected):
        return False
    if not require_density_match:
        return True
    for phase in (1, 2):
        if float(meta.get(f"phase{phase}_arrival_rate", -1)) != float(
            density[f"phase{phase}_arrival_rate"]
        ):
            return False
        for kind in ("short", "long"):
            key = f"phase{phase}_{kind}_count"
            if key in density and int(meta.get(f"phase{phase}_config_{kind}_count", -1)) != int(
                density[key]
            ):
                return False
    return True


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def completed_case_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(row.get("density") or "").strip(), str(row.get("model_key") or "").strip())
        for row in rows
        if str(row.get("status", "")).strip().lower() == "ok"
        and str(row.get("density") or "").strip()
        and str(row.get("model_key") or "").strip()
    }


def extract_summary_from_result_json(result_json: Path) -> dict[str, Any]:
    if not result_json.exists():
        return {}
    data = json.loads(result_json.read_text(encoding="utf-8"))
    phases = {name: data.get(name) or {} for name in ("phase1", "phase2", "phase12")}

    def metric(phase: str, key: str) -> float | None:
        value = phases[phase].get(key)
        return float_or_none(value.get("mean")) if isinstance(value, dict) else float_or_none(value)

    fields = {
        "phase1": (
            ("ttft_improve_mean", "ttft_improve_ratio"),
            ("wall_improve_mean", "round_wall_improve_ratio"),
            ("error_rate_mean", "error_rate"),
            ("scheduler_apply_mean", "scheduler_apply_ratio"),
            ("runtime_pressure_mean", "runtime_effective_pressure_avg"),
            ("runtime_target_fraction_mean", "runtime_target_fraction_avg"),
            ("runtime_target_chunk_mean", "runtime_target_chunk_avg"),
        ),
        "phase2": (
            ("ttft_improve_mean", "ttft_improve_ratio"),
            ("wall_improve_mean", "round_wall_improve_ratio"),
            ("slowdown_improve_mean", "slowdown_improve_ratio"),
            ("error_rate_mean", "wave_error_rate"),
            ("apply_ratio_mean", "phase2_apply_ratio"),
        ),
        "phase12": (
            ("ttft_improve_mean", "ttft_improve_ratio"),
            ("wall_improve_mean", "round_wall_improve_ratio"),
            ("slowdown_improve_mean", "slowdown_improve_ratio"),
            ("incremental_error_mean", "incremental_error_rate"),
            ("scheduler_apply_mean", "phase2_apply_ratio"),
            ("priority_lane_activations_mean", "phase2_priority_lane_activations"),
            ("priority_lane_seen_hits_mean", "phase2_priority_lane_seen_active_hits"),
            ("priority_lane_finished_hits_mean", "phase2_priority_lane_finished_active_hits"),
        ),
    }
    return {
        f"{phase}_{name}": metric(phase, source)
        for phase, entries in fields.items()
        for name, source in entries
    }


def build_dataset_source_payload(config: dict[str, Any]) -> dict[str, Any]:
    keys = ("key", "dataset_id", "split", "extractor", "streaming", "label", "role", "reason")
    return {
        "datasets": [
            {
                key: bool(item.get(key, False)) if key == "streaming" else item.get(key)
                for key in keys
            }
            for item in config.get("datasets", [])
            if isinstance(item, dict)
        ]
    }
