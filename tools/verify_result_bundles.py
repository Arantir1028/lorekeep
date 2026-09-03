"""Verify the compact result bundles tracked by the repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "results" / "result_bundles.json"


def _tree_digest(root: Path, suffixes: set[str]) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    total_bytes = 0
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        total_bytes += len(content)
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return len(files), total_bytes, digest.hexdigest()


def verify_bundles() -> list[str]:
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    suffixes = {str(value) for value in manifest["included_suffixes"]}
    failures: list[str] = []
    for bundle in manifest["bundles"]:
        key = str(bundle["key"])
        root = (ROOT / str(bundle["root"])).resolve()
        if not root.is_relative_to(ROOT) or not root.is_dir():
            failures.append(f"{key}: bundle root is missing or outside the repository: {root}")
            continue
        source_manifest = root / str(bundle["source_manifest"])
        if not source_manifest.is_file():
            failures.append(f"{key}: source manifest is missing: {source_manifest}")
        actual = _tree_digest(root, suffixes)
        expected = (
            int(bundle["file_count"]),
            int(bundle["byte_count"]),
            str(bundle["tree_sha256"]),
        )
        if actual != expected:
            failures.append(
                f"{key}: expected files/bytes/sha256={expected}, found {actual}"
            )
    return failures


def main() -> int:
    failures = verify_bundles()
    if failures:
        print("\n".join(failures))
        return 1
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"verified {len(payload['bundles'])} tracked result bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
