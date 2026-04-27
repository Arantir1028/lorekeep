"""Dynamic prefill slicer for Wave-Slice.

This module is intentionally vLLM-decoupled:
- It does not import vLLM internals.
- It only computes chunk plans and tensor views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import torch

from config import hw_config as cfg


@dataclass(frozen=True)
class SlicePlan:
    """One prefill-slice plan for a long request."""

    short_len: int
    long_total_len: int
    chunk_len: int
    long_offset: int

    @property
    def long_remaining_len(self) -> int:
        return max(0, self.long_total_len - (self.long_offset + self.chunk_len))

    @property
    def rope_start(self) -> int:
        # Absolute token position where this long chunk starts.
        return max(0, self.long_offset)

    @property
    def rope_end(self) -> int:
        return self.rope_start + self.chunk_len


class WaveBaseSlicer:
    """Dynamic bucket-aware token slicer.

    Notes:
    - This class computes slice sizes with bucket constraints.
    - It also provides view-based slicing helpers used by runtime hooks/tests.
    """

    def __init__(self, buckets: Sequence[int] | None = None):
        bucket_src = list(buckets) if buckets is not None else list(cfg.BUCKETS)
        if not bucket_src:
            raise ValueError("buckets cannot be empty")
        self.buckets = sorted({int(b) for b in bucket_src if int(b) > 0})
        if not self.buckets:
            raise ValueError("buckets must contain positive integers")

    def conservative_map_up(self, seq_len: int) -> int:
        seq_len = max(1, int(seq_len))
        for b in self.buckets:
            if b >= seq_len:
                return b
        return self.buckets[-1]

    def _conservative_map_down(self, seq_len: int) -> int:
        seq_len = max(1, int(seq_len))
        chosen = self.buckets[0]
        for b in self.buckets:
            if b <= seq_len:
                chosen = b
            else:
                break
        return chosen

    def choose_dynamic_chunk(
        self,
        *,
        short_len: int,
        long_len: int,
        scheduler: Any,
        t_wait_us: float,
        queue_length: int,
        baseline_chunk: int | None = None,
    ) -> int:
        """Select chunk length via scheduler and clamp to legal range."""
        short_len = max(1, int(short_len))
        long_len = max(1, int(long_len))
        if long_len <= short_len:
            return short_len

        # Ask scheduler first (WaveScheduler.schedule_real compatible).
        proposed = long_len
        if scheduler is not None and hasattr(scheduler, "schedule_real"):
            try:
                proposed = int(
                    scheduler.schedule_real(
                        S_s=short_len,
                        S_l=long_len,
                        t_wait_us=float(max(0.0, t_wait_us)),
                        queue_length=int(max(0, queue_length)),
                        baseline_chunk=(
                            None if baseline_chunk is None else int(max(1, baseline_chunk))
                        ),
                    )
                )
            except Exception:
                proposed = long_len

        # Enforce strict bucket and range constraints.
        proposed = max(short_len, min(proposed, long_len))
        proposed = self._conservative_map_down(proposed)
        return max(short_len, min(proposed, long_len))

    def make_plan(
        self,
        *,
        short_len: int,
        long_total_len: int,
        chunk_len: int,
        long_offset: int = 0,
    ) -> SlicePlan:
        short_len = max(1, int(short_len))
        long_total_len = max(1, int(long_total_len))
        long_offset = max(0, min(int(long_offset), long_total_len))
        max_chunk = max(1, long_total_len - long_offset)
        chunk_len = max(1, min(int(chunk_len), max_chunk))
        return SlicePlan(
            short_len=short_len,
            long_total_len=long_total_len,
            chunk_len=chunk_len,
            long_offset=long_offset,
        )

    def build_long_prefill_plan(
        self,
        *,
        short_len: int,
        long_total_len: int,
        scheduler: Any,
        t_wait_us: float,
        queue_length: int,
        start_offset: int = 0,
        baseline_chunk: int | None = None,
    ) -> list[SlicePlan]:
        """Build a full dynamic slicing plan for one long prefill request.

        The chunk length is chosen once using the current short/long
        heterogeneity signal, then expanded into offset-aware per-chunk plans.
        Downstream runtimes can either consume the whole list or just use the
        first chunk as the per-step rewrite target.
        """
        total = max(1, int(long_total_len))
        offset = max(0, min(int(start_offset), total - 1))
        remaining = total - offset
        chunk_len = self.choose_dynamic_chunk(
            short_len=short_len,
            long_len=remaining,
            scheduler=scheduler,
            t_wait_us=t_wait_us,
            queue_length=queue_length,
            baseline_chunk=baseline_chunk,
        )
        return [
            self.make_plan(
                short_len=short_len,
                long_total_len=total,
                chunk_len=chunk_len,
                long_offset=chunk_offset,
            )
            for chunk_offset, _ in self.iter_long_chunks(
                long_total_len=total,
                chunk_len=chunk_len,
                start_offset=offset,
            )
        ]

    def iter_long_chunks(
        self,
        *,
        long_total_len: int,
        chunk_len: int,
        start_offset: int = 0,
    ) -> Iterator[tuple[int, int]]:
        """Yield `(offset, length)` for chunk-by-chunk long prefill."""
        total = max(1, int(long_total_len))
        offset = max(0, min(int(start_offset), total))
        step = max(1, int(chunk_len))
        while offset < total:
            n = min(step, total - offset)
            yield offset, n
            offset += n

    @staticmethod
    def _view_2d(x: torch.Tensor, offset: int, length: int) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected 2D tensor, got ndim={x.ndim}")
        if length <= 0:
            raise ValueError("length must be positive")
        if offset < 0 or offset + length > x.shape[0]:
            raise ValueError(
                f"invalid slice offset={offset}, length={length}, size0={x.shape[0]}"
            )
        return x[offset : offset + length, :]

    def pack_short_and_long_chunk(
        self,
        *,
        x_short: torch.Tensor,
        x_long: torch.Tensor,
        plan: SlicePlan,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Create one packed tensor `[short ; long_chunk]` and metadata.

        Returns:
            packed: tensor of shape [short_len + chunk_len, hidden]
            meta: indices and offsets needed by downstream scheduling.
        """
        if x_short.ndim != 2 or x_long.ndim != 2:
            raise ValueError("x_short and x_long must be 2D")
        if x_short.shape[1] != x_long.shape[1]:
            raise ValueError("x_short and x_long hidden dims must match")
        if x_short.shape[0] < plan.short_len:
            raise ValueError("x_short shorter than plan.short_len")
        if x_long.shape[0] < plan.long_total_len:
            raise ValueError("x_long shorter than plan.long_total_len")

        short_view = self._view_2d(x_short, 0, plan.short_len)
        long_view = self._view_2d(x_long, plan.long_offset, plan.chunk_len)
        packed = torch.cat([short_view, long_view], dim=0)
        meta = {
            "short_len": int(plan.short_len),
            "chunk_len": int(plan.chunk_len),
            "long_offset": int(plan.long_offset),
            "long_remaining_len": int(plan.long_remaining_len),
            "rope_start": int(plan.rope_start),
            "rope_end": int(plan.rope_end),
        }
        return packed, meta

    def chunk_token_ids(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        offset: int,
        chunk_len: int,
    ) -> tuple[Sequence[int] | torch.Tensor, dict[str, int]]:
        """Slice token IDs without copy for list/tuple/tensor semantics."""
        offset = max(0, int(offset))
        chunk_len = max(1, int(chunk_len))
        total = len(token_ids)
        if offset >= total:
            raise ValueError("offset exceeds token length")
        end = min(total, offset + chunk_len)
        chunk = token_ids[offset:end]
        meta = {
            "offset": offset,
            "chunk_len": end - offset,
            "remaining_len": total - end,
            "rope_start": offset,
            "rope_end": end,
        }
        return chunk, meta


if __name__ == "__main__":
    # Minimal self-check.
    slicer = WaveBaseSlicer()
    x_s = torch.randn(64, 16, device="cpu")
    x_l = torch.randn(4096, 16, device="cpu")
    plan = slicer.make_plan(short_len=64, long_total_len=4096, chunk_len=1024, long_offset=2048)
    packed, meta = slicer.pack_short_and_long_chunk(x_short=x_s, x_long=x_l, plan=plan)
    print("packed.shape:", tuple(packed.shape))
    print("meta:", meta)
