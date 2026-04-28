from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from config import hw_config as hw_cfg

FINGERPRINT_SCHEMA_VERSION = 1


def _import_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    value = getattr(mod, "__version__", None)
    return str(value) if value else None


def _torch_gpu_components() -> tuple[bool, str | None, list[dict[str, Any]]]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_cuda = str(getattr(torch.version, "cuda", "") or "") or None
        gpus: list[dict[str, Any]] = []
        if cuda_available:
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(
                    {
                        "name": str(props.name),
                        "total_memory_bytes": int(props.total_memory),
                        "capability": list(torch.cuda.get_device_capability(idx)),
                    }
                )
        return cuda_available, torch_cuda, gpus
    except Exception:
        return False, None, []


def _nvidia_smi_versions() -> dict[str, str]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    for line in (proc.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            return {"driver_version": parts[0], "nvidia_smi_cuda_version": parts[1]}
    return {}


def current_lut_fingerprint(environment: dict[str, Any] | None = None) -> dict[str, Any]:
    if environment is None:
        cuda_available, torch_cuda, gpus = _torch_gpu_components()
        environment = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "torch_version": _import_version("torch"),
            "torch_cuda": torch_cuda,
            "cuda_available": cuda_available,
            "vllm_version": _import_version("vllm"),
            "gpus": gpus,
        }
        environment.update(_nvidia_smi_versions())

    components = {
        "gpus": [
            {
                "name": str(gpu.get("name") or ""),
                "total_memory_bytes": int(gpu.get("total_memory_bytes") or 0),
                "capability": list(gpu.get("capability") or []),
            }
            for gpu in list(environment.get("gpus") or [])
            if isinstance(gpu, dict)
        ],
        "torch_version": environment.get("torch_version"),
        "torch_cuda": environment.get("torch_cuda"),
        "vllm_version": environment.get("vllm_version"),
        "driver_version": environment.get("driver_version"),
        "nvidia_smi_cuda_version": environment.get("nvidia_smi_cuda_version"),
    }
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "fingerprint_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "components": components,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_lut_fingerprint(lut_name: str) -> dict[str, Any] | None:
    sanity_path = Path(hw_cfg.DATA_DIR) / f"runtime_sanity_{lut_name}.json"
    calibration_path = Path(hw_cfg.DATA_DIR) / f"runtime_calibration_{lut_name}.json"
    for path in (sanity_path, calibration_path):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        fingerprint = payload.get("hardware_fingerprint")
        if isinstance(fingerprint, dict):
            return fingerprint
    return None


def lut_fingerprint_status(
    lut_name: str,
    *,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = current_lut_fingerprint(environment)
    stored = read_lut_fingerprint(lut_name)
    if not stored:
        return {
            "ok": False,
            "reason": "missing_lut_hardware_fingerprint",
            "current": current,
            "stored": None,
        }
    current_id = str(current.get("fingerprint_id") or "")
    stored_id = str(stored.get("fingerprint_id") or "")
    ok = bool(current_id and stored_id and current_id == stored_id)
    return {
        "ok": ok,
        "reason": "" if ok else "lut_hardware_fingerprint_mismatch",
        "current": current,
        "stored": stored,
    }
