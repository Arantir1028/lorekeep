from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config import hw_config as hw_cfg
from config.experiment_catalog import ExperimentModelSpec, get_model_specs, safe_key
from experiments.lut_fingerprint import lut_fingerprint_status


@dataclass(frozen=True)
class ResolvedModel:
    key: str
    model_id: str
    lut_name: str
    trust_remote_code: bool
    max_model_len_override: Optional[int]
    model_path_mode: str
    label: str
    reason: str


def resolve_model_entry(entry: Any) -> ResolvedModel:
    if isinstance(entry, str):
        spec = get_model_specs(entry)[0]
        return ResolvedModel(
            key=spec.key,
            model_id=spec.model_id,
            lut_name=spec.lut_name,
            trust_remote_code=spec.trust_remote_code,
            max_model_len_override=spec.max_model_len_override,
            model_path_mode="local_snapshot_preferred",
            label=spec.key,
            reason="Selected from the experiment catalog.",
        )
    if not isinstance(entry, dict):
        raise ValueError(f"invalid model entry: {entry!r}")
    key = str(entry.get("key", "")).strip()
    spec: Optional[ExperimentModelSpec] = None
    if key:
        try:
            candidates = get_model_specs(key)
        except Exception:
            candidates = []
        if candidates:
            spec = candidates[0]
    model_id = str(entry.get("model_id") or (spec.model_id if spec else "")).strip()
    lut_name = str(entry.get("lut_name") or (spec.lut_name if spec else "")).strip()
    if not key:
        key = safe_key(model_id or lut_name)
    if not model_id or not lut_name:
        raise ValueError(f"model entry needs model_id and lut_name: {entry!r}")
    return ResolvedModel(
        key=key,
        model_id=model_id,
        lut_name=lut_name,
        trust_remote_code=bool(entry.get("trust_remote_code", spec.trust_remote_code if spec else False)),
        max_model_len_override=(
            int(entry["max_model_len_override"])
            if entry.get("max_model_len_override") is not None
            else (spec.max_model_len_override if spec else None)
        ),
        model_path_mode=str(entry.get("model_path_mode") or "local_snapshot_preferred"),
        label=str(entry.get("label") or key),
        reason=str(entry.get("reason") or "No rationale provided."),
    )


def runtime_lut_is_valid(lut_name: str) -> tuple[bool, str]:
    sanity_path = Path(hw_cfg.DATA_DIR) / f"runtime_sanity_{lut_name}.json"
    if not sanity_path.exists():
        return False, f"missing runtime sanity file: {sanity_path}"
    try:
        payload = json.loads(sanity_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid runtime sanity file: {exc!r}"
    if bool(payload.get("passed", False)):
        fingerprint = lut_fingerprint_status(lut_name)
        if not fingerprint.get("ok"):
            return False, str(fingerprint.get("reason") or "lut_hardware_fingerprint_mismatch")
        return True, ""
    reasons = payload.get("reasons") or []
    return False, f"runtime LUT sanity failed: {reasons}"
