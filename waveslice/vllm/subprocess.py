"""Configuration transport for vLLM child processes."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from typing import Any

from waveslice.metrics import RUNTIME_METRICS_FILE_ENV
from waveslice.policy import WaveSlicePolicy

RUNTIME_ENV_ENABLED = "WAVESLICE_ENABLED"
RUNTIME_ENV_MODEL = "WAVESLICE_MODEL_NAME"
RUNTIME_ENV_GAMMA = "WAVESLICE_GAMMA"
RUNTIME_ENV_POLICY = "WAVESLICE_POLICY_JSON"
RUNTIME_ENV_SCHEDULER = "WAVESLICE_SCHEDULER"
_PREVIOUS_PYTHONPATH = "WAVESLICE_PREVIOUS_PYTHONPATH"
_PREVIOUS_VLLM_PLUGINS = "WAVESLICE_PREVIOUS_VLLM_PLUGINS"
_PLUGIN_NAME = "waveslice"


def ensure_cross_process_metrics_file() -> str:
    path = os.environ.get(RUNTIME_METRICS_FILE_ENV, "").strip()
    if path:
        return path

    descriptor, path = tempfile.mkstemp(prefix="waveslice_metrics_", suffix=".jsonl", dir="/tmp")
    os.close(descriptor)
    os.environ[RUNTIME_METRICS_FILE_ENV] = path
    return path


def _qualified_name(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def publish_runtime_environment(
    model_name: str,
    gamma: float,
    policy: WaveSlicePolicy,
    scheduler_cls: type,
) -> None:
    """Publish the active configuration before vLLM starts child processes."""

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    previous_path = os.environ.get("PYTHONPATH", "")
    previous_plugins = os.environ.get("VLLM_PLUGINS", "")

    os.environ.update(
        {
            RUNTIME_ENV_ENABLED: "1",
            RUNTIME_ENV_MODEL: str(model_name),
            RUNTIME_ENV_GAMMA: str(float(gamma)),
            RUNTIME_ENV_POLICY: json.dumps(asdict(policy), sort_keys=True),
            RUNTIME_ENV_SCHEDULER: _qualified_name(scheduler_cls),
            RUNTIME_METRICS_FILE_ENV: ensure_cross_process_metrics_file(),
            _PREVIOUS_PYTHONPATH: previous_path,
            _PREVIOUS_VLLM_PLUGINS: previous_plugins,
        }
    )

    paths = [item for item in previous_path.split(os.pathsep) if item and item != project_root]
    os.environ["PYTHONPATH"] = os.pathsep.join([project_root, *paths])

    plugins = [item.strip() for item in previous_plugins.split(",") if item.strip()]
    if _PLUGIN_NAME not in plugins:
        plugins.append(_PLUGIN_NAME)
    os.environ["VLLM_PLUGINS"] = ",".join(plugins)


def clear_runtime_environment() -> None:
    for saved, target in (
        (_PREVIOUS_PYTHONPATH, "PYTHONPATH"),
        (_PREVIOUS_VLLM_PLUGINS, "VLLM_PLUGINS"),
    ):
        value = os.environ.pop(saved, None)
        if value:
            os.environ[target] = value
        elif value is not None:
            os.environ.pop(target, None)

    for key in (
        RUNTIME_ENV_ENABLED,
        RUNTIME_ENV_MODEL,
        RUNTIME_ENV_GAMMA,
        RUNTIME_ENV_POLICY,
        RUNTIME_ENV_SCHEDULER,
        RUNTIME_METRICS_FILE_ENV,
    ):
        os.environ.pop(key, None)


def reset_cross_process_metrics_file() -> None:
    with open(ensure_cross_process_metrics_file(), "w", encoding="utf-8"):
        pass


def read_cross_process_metrics() -> dict[str, Any]:
    path = os.environ.get(RUNTIME_METRICS_FILE_ENV, "").strip()
    merged: dict[str, Any] = {
        "values": {},
        "reasons": {},
        "last_active_ids": [],
        "last_deferred_ids": [],
    }
    if not path or not os.path.exists(path):
        return merged

    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    for record in records:
        if int(record["pid"]) == os.getpid():
            continue
        payload = record["payload"]
        for name, value in payload["values"].items():
            merged["values"][name] = merged["values"].get(name, 0) + value
        if reason := payload.get("reason"):
            merged["reasons"][reason] = merged["reasons"].get(reason, 0) + 1
        if "last_active_ids" in payload:
            merged["last_active_ids"] = payload["last_active_ids"]
            merged["last_deferred_ids"] = payload["last_deferred_ids"]
    return merged


__all__ = [
    "RUNTIME_ENV_ENABLED",
    "RUNTIME_ENV_GAMMA",
    "RUNTIME_ENV_MODEL",
    "RUNTIME_ENV_POLICY",
    "RUNTIME_ENV_SCHEDULER",
    "clear_runtime_environment",
    "publish_runtime_environment",
    "read_cross_process_metrics",
    "reset_cross_process_metrics_file",
]
