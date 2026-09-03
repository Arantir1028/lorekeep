from __future__ import annotations

import contextlib
import fcntl
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

DEFAULT_GPU_LOCK_PATH = Path(
    os.environ.get(
        "WAVESLICE_GPU_LOCK_PATH",
        str(Path(tempfile.gettempdir()) / "waveslice_gpu_experiment.lock"),
    )
)


@contextlib.contextmanager
def gpu_experiment_lock(
    *,
    label: str,
    enabled: bool = True,
    lock_path: str | None = None,
    poll_interval_s: float = 2.0,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    path = Path(lock_path) if lock_path else DEFAULT_GPU_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    announced_wait = False
    with open(path, "a+", encoding="utf-8") as fd:
        try:
            while True:
                try:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not announced_wait:
                        print(
                            f"[WaveSlice-Lock] waiting for GPU experiment lock: {path} "
                            f"(label={label})",
                            flush=True,
                        )
                        announced_wait = True
                    time.sleep(max(0.1, float(poll_interval_s)))
            fd.seek(0)
            fd.truncate()
            fd.write(
                f"pid={os.getpid()}\nlabel={label}\ncwd={os.getcwd()}\nargv={' '.join(sys.argv)}\n"
            )
            fd.flush()
            os.fsync(fd.fileno())
            print(
                f"[WaveSlice-Lock] acquired GPU experiment lock: {path} (label={label})",
                flush=True,
            )
            yield
        finally:
            fd.seek(0)
            fd.truncate()
            fd.flush()
            os.fsync(fd.fileno())
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
