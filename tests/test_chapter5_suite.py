from __future__ import annotations

from experiments.run_chapter5_suite import _resolve_stages


def test_resolve_stages_accepts_all_and_explicit_subset() -> None:
    assert _resolve_stages("all") == ["main", "baseline", "figures", "partial-figures"]
    assert _resolve_stages("main,figures") == ["main", "figures"]
