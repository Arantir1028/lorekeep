import unittest

from scheduler.wave_scheduler import WaveScheduler


class SchedulerGridTests(unittest.TestCase):
    def test_schedule_matrix_matches_expected_edge_cases(self) -> None:
        scheduler = WaveScheduler(gamma=0.5, max_queue_depth=100)
        expected = {
            (0.0, 0): 1536,
            (0.0, 25): 2048,
            (50.0, 0): 1536,
            (50.0, 100): 1536,
            (500.0, 100): 1536,
        }
        for (wait_us, queue_depth), target in expected.items():
            with self.subTest(wait_us=wait_us, queue_depth=queue_depth):
                best_sc = scheduler.schedule(
                    S_s=45,
                    S_l=2048,
                    t_wait_s_us=wait_us,
                    t_solo_s_us=50.0,
                    current_queue_depth=queue_depth,
                )
                self.assertEqual(best_sc, target)

    def test_schedule_real_respects_legal_bounds(self) -> None:
        scheduler = WaveScheduler(gamma=0.5, max_queue_depth=100)
        for wait_us in [0.0, 50.0, 100.0, 200.0, 500.0]:
            for queue_depth in [0, 25, 50, 75, 100]:
                with self.subTest(wait_us=wait_us, queue_depth=queue_depth):
                    best_sc = scheduler.schedule_real(
                        S_s=45,
                        S_l=2048,
                        t_wait_us=wait_us,
                        queue_length=queue_depth,
                    )
                    self.assertGreaterEqual(best_sc, 45)
                    self.assertLessEqual(best_sc, 2048)


if __name__ == "__main__":
    unittest.main()
