"""Check repository-local links in maintained Markdown files."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _markdown_files() -> list[Path]:
    commands = (
        ["git", "ls-files", "-z", "--", "*.md"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "*.md"],
    )
    paths: set[Path] = set()
    for command in commands:
        output = subprocess.run(command, cwd=ROOT, check=True, capture_output=True).stdout
        paths.update(ROOT / value.decode("utf-8") for value in output.split(b"\0") if value)
    return sorted(path for path in paths if path.is_file())


def _local_target(raw: str) -> str | None:
    value = raw.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    else:
        try:
            value = shlex.split(value)[0]
        except (ValueError, IndexError):
            return None
    if not value or value.startswith("#") or "://" in value or value.startswith("mailto:"):
        return None
    return unquote(value.split("#", 1)[0])


def broken_links() -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []
    for path in _markdown_files():
        for raw in LINK.findall(path.read_text(encoding="utf-8")):
            target = _local_target(raw)
            if target is None:
                continue
            checked += 1
            resolved = (ROOT / target.lstrip("/")) if target.startswith("/") else path.parent / target
            if not resolved.exists():
                failures.append(f"{path.relative_to(ROOT)}: missing link target: {target}")
    return checked, failures


def main() -> int:
    checked, failures = broken_links()
    if failures:
        print("\n".join(failures))
        return 1
    print(f"checked {checked} local Markdown links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
