from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.openworkload_support import load_config, repo_root
from waveslice.policy import WaveSlicePolicy

EVAL_CONFIG_ENV = "WAVESLICE_EVAL_CONFIG_JSON"
_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "purpose",
    "evaluator",
    "result_json",
    "runtime",
    "phase1",
    "phase12_soft_gate",
    "phase2",
    "model",
    "workload",
    "adapters",
    "include_phase12",
    "skip_phase2",
    "baseline_only",
    "ignore_eos",
    "short_repeat",
    "short_a_repeat",
    "short_b_repeat",
    "long_repeat",
}
_RUNTIME_KEYS = {
    "python_bin",
    "trust_remote_code",
    "warmup_iters",
    "repeats",
    "timeout_sec",
    "max_new_tokens",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_partial_prefills",
    "max_long_partial_prefills",
    "gpu_memory_utilization",
    "queue_reorder_mode",
    "queue_reorder_aging_quantum_us",
    "adapters_root",
}
_POLICY_KEYS = set(WaveSlicePolicy.__dataclass_fields__)
_SECTION_KEYS = {
    "runtime": _RUNTIME_KEYS,
    "phase1": {name.removeprefix("phase1_") for name in _POLICY_KEYS if name.startswith("phase1_")}
    | {"objective_mode", "baseline_mode", "gamma"},
    "phase2": {name.removeprefix("phase2_") for name in _POLICY_KEYS if name.startswith("phase2_")}
    | {"dispatch_mode", "baseline_enable_chunked_prefill", "enable_mixed_prefill_decode"},
    "phase12_soft_gate": {
        name.removeprefix("phase12_phase2_")
        for name in _POLICY_KEYS
        if name.startswith("phase12_phase2_")
    }
    | {"phase2_gate_mode"},
    "model": {"name", "path"},
    "workload": {"requests_json", "lora_requests_json"},
    "adapters": {"adapter_a", "adapter_b", "auto_build"},
}


def eval_section_keys(section: str) -> frozenset[str]:
    return frozenset(_SECTION_KEYS[section])


def validate_eval_config(config: dict[str, Any]) -> None:
    unknown = set(config) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown evaluation config keys: {sorted(unknown)}")
    for section, allowed in _SECTION_KEYS.items():
        values = config.get(section) or {}
        if not isinstance(values, dict):
            raise TypeError(f"evaluation config section {section!r} must be an object")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown evaluation config keys in {section}: {sorted(unknown)}")


def apply_eval_config(args: Any, config: dict[str, Any]) -> Any:
    validate_eval_config(config)
    for section, prefix in (("runtime", ""), ("phase1", "phase1_"), ("phase2", "phase2_")):
        for key, value in dict(config.get(section) or {}).items():
            setattr(args, prefix + key, value)
    for key, value in dict(config.get("phase12_soft_gate") or {}).items():
        prefix = "phase12_" if key.startswith("phase2_") else "phase12_phase2_"
        setattr(args, prefix + key, value)
    for key, value in dict(config.get("model") or {}).items():
        setattr(args, {"name": "model_name", "path": "model_path"}[key], value)
    for section in ("workload", "adapters"):
        for key, value in dict(config.get(section) or {}).items():
            setattr(args, key, value)
    for key in (
        "include_phase12",
        "skip_phase2",
        "baseline_only",
        "ignore_eos",
        "short_repeat",
        "short_a_repeat",
        "short_b_repeat",
        "long_repeat",
    ):
        if key in config:
            setattr(args, key, config[key])
    return args


def build_eval_invocation(
    config: dict[str, Any], *, out_json_override: str | None = None
) -> tuple[list[str], dict[str, str]]:
    validate_eval_config(config)
    runtime = dict(config.get("runtime") or {})
    evaluator = str(config.get("evaluator") or "tests/evaluate_waveslice_claims.py")
    command = [str(runtime.get("python_bin") or sys.executable), evaluator]
    result = out_json_override or str(config.get("result_json") or "")
    if result:
        command.extend(["--out-json", str(result)])
    env = os.environ.copy()
    env.update(
        {
            EVAL_CONFIG_ENV: json.dumps(config, ensure_ascii=False),
            "VLLM_USE_V1": "1",
            "VLLM_NO_USAGE_STATS": "1",
        }
    )
    root = str(repo_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return command, env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen WaveSlice evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    command, env = build_eval_invocation(config, out_json_override=args.out_json or None)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    print("[FrozenEval] " + shlex.join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command, env=env, cwd=repo_root(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
