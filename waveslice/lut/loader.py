"""Load the three scheduler lookup tables."""

import json
from pathlib import Path

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


def load_model_luts(model_name: str):
    paths = cfg.get_lut_paths(model_name)
    missing = [path for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing WaveSlice LUT files for model={model_name}: {', '.join(missing)}"
        )
    return load_lut_triplet(paths)
