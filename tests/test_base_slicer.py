from __future__ import annotations

import unittest

import torch

from engine.base_slicer import WaveBaseSlicer


class _DummyScheduler:
    def schedule_real(
        self,
        S_s: int,
        S_l: int,
        t_wait_us: float,
        queue_length: int,
        baseline_chunk: int | None = None,
    ) -> int:
        # Return a non-bucket value on purpose; slicer should clamp to bucket.
        _ = baseline_chunk
        return max(S_s, min(S_l - 1, 1500))


class BaseSlicerTests(unittest.TestCase):
    def test_dynamic_chunk_clamped_to_bucket(self) -> None:
        slicer = WaveBaseSlicer(buckets=[32, 64, 128, 256, 512, 1024, 2048, 4096])
        chosen = slicer.choose_dynamic_chunk(
            short_len=100,
            long_len=3000,
            scheduler=_DummyScheduler(),
            t_wait_us=1000.0,
            queue_length=8,
        )
        self.assertEqual(chosen, 1024)

    def test_pack_short_and_long_chunk_meta(self) -> None:
        slicer = WaveBaseSlicer()
        x_s = torch.randn(64, 32)
        x_l = torch.randn(512, 32)
        plan = slicer.make_plan(short_len=64, long_total_len=512, chunk_len=128, long_offset=256)
        packed, meta = slicer.pack_short_and_long_chunk(x_short=x_s, x_long=x_l, plan=plan)
        self.assertEqual(tuple(packed.shape), (192, 32))
        self.assertEqual(meta["long_offset"], 256)
        self.assertEqual(meta["rope_start"], 256)
        self.assertEqual(meta["rope_end"], 384)

    def test_build_long_prefill_plan_tracks_offsets(self) -> None:
        slicer = WaveBaseSlicer(buckets=[32, 64, 128, 256, 512, 1024, 2048, 4096])
        plans = slicer.build_long_prefill_plan(
            short_len=96,
            long_total_len=1500,
            scheduler=_DummyScheduler(),
            t_wait_us=800.0,
            queue_length=4,
        )
        self.assertEqual([p.long_offset for p in plans], [0, 1024])
        self.assertEqual([p.chunk_len for p in plans], [1024, 476])
        self.assertEqual(plans[0].rope_start, 0)
        self.assertEqual(plans[1].rope_start, 1024)

    def test_chunk_token_ids_reports_rope_offsets(self) -> None:
        slicer = WaveBaseSlicer()
        chunk, meta = slicer.chunk_token_ids(list(range(32)), offset=10, chunk_len=7)
        self.assertEqual(list(chunk), [10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(
            meta,
            {
                "offset": 10,
                "chunk_len": 7,
                "remaining_len": 15,
                "rope_start": 10,
                "rope_end": 17,
            },
        )


if __name__ == "__main__":
    unittest.main()
