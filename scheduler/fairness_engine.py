# lorekeep/scheduler/fairness_engine.py

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
        """
        公平性大脑与负载评估器。
        
        参数:
            alpha: 基础敏感度乘子
            beta: 紧急度导数 (SLA 崖壁指数)
            w_max: 硬件物理熔断上限，防止无底线的切分导致系统崩溃
        """
        self.alpha = alpha
        self.beta = beta
        self.w_max = w_max
        self.rho_bypass = max(0.0, min(0.999, rho_bypass))
        self.ema_decay = max(0.0, min(1.0, float(ema_decay)))
        self.current_rho_ema = 0.0

    def compute_weight(self, t_wait_us: float, t_solo_us: float) -> float:
        """
        计算瞬时非线性公平性权重 (W_fairness)
        """
        if t_solo_us <= 0:
            return 1.0
        
        relative_starvation = t_wait_us / t_solo_us
        w = 1.0 + self.alpha * (relative_starvation ** self.beta)
        
        return min(self.w_max, w)

    def compute_rho_md1(self, queue_length: int) -> float:
        """
        【学术护城河】：基于排队论 M/D/1 稳态分布严格推导系统瞬时拥塞度 rho。
        消除所有 Magic Number。
        
        公式来源 (Pollaczek-Khinchine Formula for M/D/1): 
        L = rho + rho^2 / (2 * (1 - rho))
        解得严格物理映射: rho = (L + 1) - sqrt(L^2 + 1)
        """
        if queue_length <= 0:
            raw_rho = 0.0
        else:
            L = float(queue_length)
            raw_rho = (L + 1.0) - math.sqrt(L**2 + 1.0)
            raw_rho = min(0.99, max(0.0, raw_rho))

        if self.current_rho_ema == 0.0:
            self.current_rho_ema = raw_rho
        else:
            self.current_rho_ema = (
                self.ema_decay * raw_rho
                + (1.0 - self.ema_decay) * self.current_rho_ema
            )

        return self.current_rho_ema

    def should_elastic_bypass(self, rho: float) -> bool:
        return float(rho) >= self.rho_bypass
