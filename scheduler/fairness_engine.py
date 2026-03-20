# lorekeep/scheduler/fairness_engine.py

class FairnessEngine:
    def __init__(self, alpha=0.5, beta=2.0, w_max=10.0):
        """
        初始化公平性权重引擎。
        参数:
            alpha: 基础敏感度乘子。控制短任务触发切分的起始门槛。
            beta: 紧急度导数 (SLA 崖壁)。Beta > 1 会产生非线性惩罚，越等越急。
            w_max: 硬件熔断上限。绝对物理防线，防止为救短任务导致显存总线雪崩。
        """
        self.alpha = alpha
        self.beta = beta
        self.w_max = w_max

    def compute_weight(self, t_wait_us: float, t_solo_us: float) -> float:
        """
        计算瞬时公平性权重 W_fairness。
        时间单位统一为微秒 (us)。
        """
        # 防御性编程：避免除零错误
        if t_solo_us <= 0:
            return 1.0

        # 计算瞬时相对饥饿度
        relative_starvation = t_wait_us / t_solo_us
        
        # 核心非线性 SLA 惩罚公式
        w = 1.0 + self.alpha * (relative_starvation ** self.beta)
        
        # 触碰硬件物理防线，强制截断
        return min(self.w_max, w)

# 单元测试与验证
if __name__ == "__main__":
    engine = FairnessEngine(alpha=0.5, beta=2.0, w_max=10.0)
    # 模拟一个理想独占时间为 50us 的短任务
    t_solo = 50.0 
    print(f"{'等待时间 (us)':<15} | {'相对饥饿度':<15} | {'W_fairness 权重':<15}")
    print("-" * 50)
    for t_wait in [0, 25, 50, 100, 200, 500]:
        w = engine.compute_weight(t_wait, t_solo)
        print(f"{t_wait:<15} | {t_wait/t_solo:<15.1f} | {w:<15.2f}")