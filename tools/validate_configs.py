"""Validate maintained experiment configurations against their JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments" / "configs"
SCHEMA_PATH = ROOT / "experiments" / "schemas" / "experiment-config.schema.json"


def _format_path(parts: Any) -> str:
    return ".".join(str(part) for part in parts) or "<root>"


def _referenced_config_paths(payload: dict[str, Any]) -> list[str]:
    keys = ("resource_catalog_config", "main_config", "baseline_config")
    return [str(payload[key]) for key in keys if str(payload.get(key) or "").strip()]


def _validators_by_version(schema: dict[str, Any]) -> dict[str, Draft202012Validator]:
    validators = {}
    for name, definition in schema["$defs"].items():
        version = definition.get("properties", {}).get("schema_version", {}).get("const")
        if version:
            validators[str(version)] = Draft202012Validator(
                {
                    "$schema": schema["$schema"],
                    "$defs": schema["$defs"],
                    "$ref": f"#/$defs/{name}",
                }
            )
    return validators


def validate_configs() -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validators = _validators_by_version(schema)
    failures: list[str] = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = str(payload.get("schema_version") or "")
        validator = validators.get(version)
        if validator is None:
            failures.append(
                f"{path.relative_to(ROOT)}:<root>: unknown schema_version: {version or '<missing>'}"
            )
            continue
        errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
        for error in errors:
            failures.append(f"{path.relative_to(ROOT)}:{_format_path(error.path)}: {error.message}")
        for value in _referenced_config_paths(payload):
            referenced = ROOT / value
            if not referenced.is_file():
                failures.append(
                    f"{path.relative_to(ROOT)}:<root>: referenced config does not exist: {value}"
                )
    return failures


def main() -> int:
    failures = validate_configs()
    if failures:
        print("\n".join(failures))
        return 1
    count = len(list(CONFIG_DIR.glob("*.json")))
    print(f"validated {count} experiment configurations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
