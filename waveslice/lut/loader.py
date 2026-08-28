"""Load the three scheduler lookup tables."""

import json
import os

from waveslice.lut import config as cfg


def _json(path: str):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load_lut_triplet(paths: dict[str, str]):
    raw = {int(key): float(value) for key, value in _json(paths["raw"])["T_solo"].items()}

    def nested(path: str) -> dict[int, dict[int, float]]:
        return {
            int(row): {int(col): float(value) for col, value in values.items()}
            for row, values in _json(path).items()
        }

    return raw, nested(paths["gain"]), nested(paths["penalty"])


def legacy_lut_paths() -> dict[str, str]:
    return {
        name: os.path.join(cfg.DATA_DIR, filename)
        for name, filename in {
            "raw": "raw_profile.json",
            "gain": "lut_gain.json",
            "penalty": "lut_penalty.json",
        }.items()
    }


def load_model_luts(model_name: str):
    try:
        return load_lut_triplet(cfg.get_lut_paths(model_name))
    except FileNotFoundError:
        try:
            return load_lut_triplet(legacy_lut_paths())
        except FileNotFoundError as error:
            raise RuntimeError(f"missing LUT/profile files for model={model_name}") from error
