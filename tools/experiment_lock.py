from __future__ import annotations

import contextlib
import fcntl
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, TextIO


DEFAULT_GPU_LOCK_PATH = Path(os.environ.get("WAVESLICE_GPU_LOCK_PATH", "/tmp/waveslice_gpu_experiment.lock"))


@contextlib.contextmanager
def gpu_experiment_lock(
    *,
    label: str,
    enabled: bool = True,
    lock_path: Optional[str] = None,
    poll_interval_s: float = 2.0,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    path = Path(lock_path) if lock_path else DEFAULT_GPU_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fd: Optional[TextIO] = None
    announced_wait = False
    try:
        fd = open(path, "a+", encoding="utf-8")
        while True:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not announced_wait:
                    print(
                        f"[Wave-Slice-Lock] waiting for GPU experiment lock: {path} "
                        f"(label={label})",
                        flush=True,
                    )
                    announced_wait = True
                time.sleep(max(0.1, float(poll_interval_s)))

        fd.seek(0)
        fd.truncate()
        fd.write(
            f"pid={os.getpid()}\n"
            f"label={label}\n"
            f"cwd={os.getcwd()}\n"
            f"argv={' '.join(sys.argv)}\n"
        )
        fd.flush()
        os.fsync(fd.fileno())
        print(f"[Wave-Slice-Lock] acquired GPU experiment lock: {path} (label={label})", flush=True)
        yield
    finally:
        if fd is not None:
            try:
                fd.seek(0)
                fd.truncate()
                fd.flush()
                os.fsync(fd.fileno())
            except Exception:
                pass
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            fd.close()
