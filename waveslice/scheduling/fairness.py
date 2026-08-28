"""Fairness and queue-pressure calculations for chunk selection."""

import math


class FairnessEngine:
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 2.0,
        w_max: float = 10.0,
        rho_bypass: float = 0.98,
        ema_decay: float = 0.3,
    ):
        self.alpha, self.beta, self.w_max = alpha, beta, w_max
        self.rho_bypass = max(0.0, min(0.999, rho_bypass))
        self.ema_decay = max(0.0, min(1.0, float(ema_decay)))
        self.current_rho_ema = 0.0

    def compute_weight(self, t_wait_us: float, t_solo_us: float) -> float:
        if t_solo_us <= 0:
            return 1.0
        return min(self.w_max, 1.0 + self.alpha * (t_wait_us / t_solo_us) ** self.beta)

    def compute_rho_md1(self, queue_length: int) -> float:
        length = max(0.0, float(queue_length))
        rho = min(0.99, max(0.0, (length + 1.0) - math.sqrt(length * length + 1.0)))
        self.current_rho_ema = (
            rho
            if self.current_rho_ema == 0.0
            else self.ema_decay * rho + (1 - self.ema_decay) * self.current_rho_ema
        )
        return self.current_rho_ema

    def should_elastic_bypass(self, rho: float) -> bool:
        return rho >= self.rho_bypass
