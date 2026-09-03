"""LUT-backed WaveSlice chunk decision engine."""

from typing import Any

from waveslice.lut import config as cfg
from waveslice.lut.loader import load_model_luts
from waveslice.scheduling.fairness import FairnessEngine


class WaveScheduler:
    def __init__(
        self,
        model_name: str,
        gamma: float = 2.0,
        objective_mode: str = "fair_escape",
        **_: Any,
    ):
        self.model_name = cfg.resolve_model_name(model_name)
        self.gamma, self.objective_mode = gamma, objective_mode
        self.buckets = sorted(cfg.BUCKETS)
        self.fairness_engine = FairnessEngine()
        self.t_solo_dict, self.lut_gain, self.lut_penalty = load_model_luts(self.model_name)

    @staticmethod
    def _next(keys, target: int) -> int:
        ordered = sorted(int(key) for key in keys)
        return next((key for key in ordered if key >= target), ordered[-1])

    def _bucket(self, length: int) -> int:
        return self._next(self.buckets, length)

    _conservative_map_up = _bucket

    def _lookup(self, table, row: int, col: int | None = None) -> float:
        if not table:
            return 0.0
        row = self._next(table, row)
        if col is None:
            return float(table[row])
        values = table[row]
        return float(values[self._next(values, col)]) if values else 0.0

    def _utility(self, short: int, long: int, chunk: int, solo_long: float) -> tuple[float, float]:
        gain = self._lookup(self.lut_gain, short, chunk)
        penalty = self._lookup(self.lut_penalty, long, chunk)
        return max(0.0, solo_long - gain), penalty

    def schedule_real(
        self,
        S_s: int,
        S_l: int,
        t_wait_us: float,
        queue_length: int,
        baseline_chunk: int | None = None,
    ) -> int:
        short, long = self._bucket(S_s), self._bucket(S_l)
        solo_short, solo_long = (
            self._lookup(self.t_solo_dict, short),
            self._lookup(self.t_solo_dict, long),
        )
        rho = self.fairness_engine.compute_rho_md1(queue_length)
        if self.fairness_engine.should_elastic_bypass(rho):
            return int(baseline_chunk) if baseline_chunk is not None else S_l
        weight = self.fairness_engine.compute_weight(t_wait_us, solo_short)
        best = int(baseline_chunk) if baseline_chunk is not None else S_l
        reference = None
        if baseline_chunk is not None:
            baseline_chunk = max(1, min(int(baseline_chunk), int(S_l)))
            baseline_bucket = self._bucket(baseline_chunk)
            if baseline_chunk < S_l and baseline_bucket < long:
                reference = (
                    *self._utility(short, long, baseline_bucket, solo_long),
                    baseline_bucket,
                )
            else:
                best = S_l
        candidates = [
            chunk
            for chunk in self.buckets
            if short <= chunk < long and (reference is None or chunk < reference[2])
        ]
        best_score = 0.0
        for chunk in candidates:
            utility, penalty = self._utility(short, long, chunk, solo_long)
            if reference:
                utility, penalty = utility - reference[0], penalty - reference[1]
            score = weight * utility - penalty * (1.0 + self.gamma * rho)
            if score > best_score:
                best, best_score = chunk, score
        return best

    def schedule(self, S_s: int, S_l: int, *args: Any, **kwargs: Any) -> int:
        wait = float(kwargs.get("t_wait_us", kwargs.get("t_wait_s_us", 0.0)))
        depth = kwargs.get("queue_length", kwargs.get("current_queue_depth"))
        baseline = kwargs.get("baseline_chunk")
        if depth is None and len(args) >= 4:
            wait, rho = float(args[2]), max(0.0, min(0.99, float(args[3])))
            depth = 0 if rho <= 0 else max(1, round(rho + rho * rho / (2 * (1 - rho))))
        elif depth is None and len(args) >= 2:
            wait, depth = float(args[0]), int(args[1])
        return self.schedule_real(
            S_s, S_l, wait, int(depth or 0), None if baseline is None else int(baseline)
        )
