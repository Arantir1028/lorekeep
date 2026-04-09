"""LUT/profile loading helpers for WaveScheduler."""

from __future__ import annotations

import json
import os
from typing import Any

from config import hw_config as cfg


def _load_raw_profile(path: str) -> dict[int, float]:
    with open(path, "r", encoding="utf-8") as fh:
        raw_data = json.load(fh)
    return {int(k): float(v) for k, v in raw_data["T_solo"].items()}


def _load_nested_lut(path: str) -> dict[int, dict[int, float]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)
    return {
        int(k): {int(kk): float(vv) for kk, vv in v.items()}
        for k, v in payload.items()
    }


def legacy_lut_paths() -> dict[str, str]:
    return {
        "raw": os.path.join(cfg.DATA_DIR, "raw_profile.json"),
        "gain": os.path.join(cfg.DATA_DIR, "lut_gain.json"),
        "penalty": os.path.join(cfg.DATA_DIR, "lut_penalty.json"),
    }


def load_lut_triplet(paths: dict[str, str]) -> tuple[dict[int, float], dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    return (
        _load_raw_profile(paths["raw"]),
        _load_nested_lut(paths["gain"]),
        _load_nested_lut(paths["penalty"]),
    )


def load_model_luts(model_name: str) -> tuple[dict[int, float], dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    paths = cfg.get_lut_paths(model_name)
    try:
        return load_lut_triplet(paths)
    except FileNotFoundError:
        try:
            return load_lut_triplet(legacy_lut_paths())
        except FileNotFoundError as legacy_exc:
            raise RuntimeError(
                f"Fatal Error: missing LUT/profile files for model={model_name}. "
                "Please run offline profiler first."
            ) from legacy_exc
