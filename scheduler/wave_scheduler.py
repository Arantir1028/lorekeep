# lorekeep/scheduler/wave_scheduler.py

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg
from scheduler.fairness_engine import FairnessEngine

class WaveScheduler:
    def __init__(self, model_name: str, gamma: float = 2.0):
        """
        纯物理决断引擎。
        无任何先验经验数值，所有决策严格依托 GPU 底层实测张量时间。
        
        参数:
            model_name: 模型名称
            gamma: 惩罚放大因子。控制负载对决策的敏感度。
        """
        self.model_name = model_name
        self.gamma = gamma
        self.buckets = sorted(cfg.BUCKETS)
        self.fairness_engine = FairnessEngine()  # 引入公平性大脑
        self._load_luts()

    def _load_luts(self):
        paths = cfg.get_lut_paths(self.model_name)
        try:
            with open(paths["gain"], "r") as f:
                self.lut_gain = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(f).items()}
            with open(paths["penalty"], "r") as f:
                self.lut_penalty = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(f).items()}
        except FileNotFoundError:
            raise Exception(f"Fatal Error: 找不到 {self.model_name} 的 O(1) 决断表。请先运行 lut_generator.py。")

    def _conservative_map_up(self, seq_len: int) -> int:
        """保守映射：确保物理预估绝对处于安全边界内 (低估收益，高估惩罚)"""
        for b in self.buckets:
            if b >= seq_len:
                return b
        return self.buckets[-1]

    def schedule(self, S_s: int, S_l: int, t_solo_s: float, t_solo_l: float, t_wait_s: float, rho: float) -> int:
        """
        基于全局 Slowdown 守恒定律与 SLA 公平性的 O(1) 物理决断核心。
        
        参数:
            S_s, S_l: 长短任务的序列长度
            t_solo_s, t_solo_l: 长短任务的理论基线独占执行时间
            t_wait_s: 短任务已经历的真实等待时间
            rho: 瞬时系统负载饱和度系数 [0, 1)
        返回:
            最优的 Chunk 切分粒度 S_c。若所有决断收益 <= 0，则等于 S_l 触发熔断。
        """
        b_s = self._conservative_map_up(S_s)
        b_l = self._conservative_map_up(S_l)

        best_S_c = S_l   
        max_net_benefit = 0.0  

        valid_chunk_candidates = [b for b in self.buckets if b_s <= b < b_l]
        
        # 1. 计算瞬时公平性杠杆 W_fairness
        w_fairness = self.fairness_engine.compute_weight(t_wait_s, t_solo_s)

        for S_c in valid_chunk_candidates:
            # O(1) 获取真实物理耗时
            t_conc_s = self.lut_gain[b_s][S_c]
            t_penalty = self.lut_penalty[b_l][S_c]

            # 2. 拯救短任务的绝对时间收益 (Delta U)
            # 物理意义：如果不切分，短任务被强制阻塞 t_solo_l；切分后只等待 t_conc_s
            delta_u = t_solo_l - t_conc_s
            
            # 3. 拥塞感知的系统物理惩罚
            # 物理意义：系统越忙碌(rho 越高)，基础物理惩罚对全局吞吐量的破坏性越强
            cost_global = t_penalty * (1.0 + self.gamma * rho)

            # 4. 核心决断不等式
            net_benefit = w_fairness * delta_u - cost_global

            if net_benefit > max_net_benefit:
                max_net_benefit = net_benefit
                best_S_c = S_c

        return best_S_c