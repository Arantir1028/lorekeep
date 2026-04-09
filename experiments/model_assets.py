from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from config.experiment_catalog import DEFAULT_SYNTHETIC_ADAPTER_PRESETS
from tools.synthetic_lora_builder import AdapterSpec, build_synthetic_adapters


def resolve_local_snapshot(model_id: str) -> Optional[str]:
    home = Path.home()
    hub_dir = home / ".cache" / "huggingface" / "hub"
    repo_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = hub_dir / repo_name / "snapshots"
    if not snapshots_dir.exists():
        return None
    dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in dirs:
        if (
            (snap / "config.json").exists()
            and (
                (snap / "tokenizer_config.json").exists()
                or (snap / "tokenizer.json").exists()
                or (snap / "vocab.json").exists()
            )
        ):
            return str(snap)
    for snap in dirs:
        if (snap / "config.json").exists():
            return str(snap)
    return None


def ensure_adapters(
    *,
    base_model_path: str,
    out_dir: str,
    trust_remote_code: bool,
) -> tuple[str, str]:
    preset_a, preset_b = DEFAULT_SYNTHETIC_ADAPTER_PRESETS[:2]
    path_a = os.path.join(out_dir, preset_a.name)
    path_b = os.path.join(out_dir, preset_b.name)
    marker_a = os.path.join(path_a, "adapter_config.json")
    marker_b = os.path.join(path_b, "adapter_config.json")
    if os.path.exists(marker_a) and os.path.exists(marker_b):
        return path_a, path_b

    generated = build_synthetic_adapters(
        base_model=base_model_path,
        out_dir=out_dir,
        specs=[
            AdapterSpec(
                name=spec.name,
                rank=spec.rank,
                alpha=spec.alpha,
                seed=spec.seed,
                init_std=spec.init_std,
            )
            for spec in DEFAULT_SYNTHETIC_ADAPTER_PRESETS
        ],
        trust_remote_code=trust_remote_code,
    )
    return generated[0], generated[1]
