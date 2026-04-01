from __future__ import annotations

try:
    from engine.vllm_hijacker import _maybe_auto_inject_from_env

    _maybe_auto_inject_from_env()
except Exception:
    # Never let sitecustomize break interpreter startup.
    pass
