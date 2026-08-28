from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from waveslice.lut import config as cfg


@dataclass(frozen=True)
class SlicePlan:
    short_len: int
    long_total_len: int
    chunk_len: int
    long_offset: int


class WaveBaseSlicer:
    def __init__(self, buckets: Sequence[int] | None = None):
        self.buckets = sorted({int(value) for value in (buckets or cfg.BUCKETS) if int(value) > 0})
        if not self.buckets:
            raise ValueError("buckets must contain a positive integer")

    def _conservative_map_down(self, length: int) -> int:
        eligible = [bucket for bucket in self.buckets if bucket <= max(1, int(length))]
        return eligible[-1] if eligible else self.buckets[0]

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
        short_len, long_len = max(1, int(short_len)), max(1, int(long_len))
        if long_len <= short_len:
            return short_len
        proposed = (
            scheduler.schedule_real(
                S_s=short_len,
                S_l=long_len,
                t_wait_us=max(0.0, float(t_wait_us)),
                queue_length=max(0, int(queue_length)),
                baseline_chunk=None if baseline_chunk is None else max(1, int(baseline_chunk)),
            )
            if scheduler is not None
            else long_len
        )
        return max(short_len, min(self._conservative_map_down(int(proposed)), long_len))

    @staticmethod
    def make_plan(
        *, short_len: int, long_total_len: int, chunk_len: int, long_offset: int = 0
    ) -> SlicePlan:
        total = max(1, int(long_total_len))
        offset = max(0, min(int(long_offset), total))
        return SlicePlan(
            max(1, int(short_len)),
            total,
            max(1, min(int(chunk_len), max(1, total - offset))),
            offset,
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
        total = max(1, int(long_total_len))
        offset = max(0, min(int(start_offset), total - 1))
        chunk = self.choose_dynamic_chunk(
            short_len=short_len,
            long_len=total - offset,
            scheduler=scheduler,
            t_wait_us=t_wait_us,
            queue_length=queue_length,
            baseline_chunk=baseline_chunk,
        )
        return [
            self.make_plan(
                short_len=short_len, long_total_len=total, chunk_len=chunk, long_offset=position
            )
            for position, _ in self.iter_long_chunks(
                long_total_len=total, chunk_len=chunk, start_offset=offset
            )
        ]

    @staticmethod
    def iter_long_chunks(
        *, long_total_len: int, chunk_len: int, start_offset: int = 0
    ) -> Iterator[tuple[int, int]]:
        total, step = max(1, int(long_total_len)), max(1, int(chunk_len))
        offset = max(0, min(int(start_offset), total))
        while offset < total:
            size = min(step, total - offset)
            yield offset, size
            offset += size
