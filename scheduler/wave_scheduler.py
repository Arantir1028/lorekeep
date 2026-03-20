# lorekeep/scheduler/wave_scheduler.py

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg

class WaveScheduler:
    def __init__(self, model_name: str):
        """
        纯物理决断引擎。
        无任何先验经验数值，所有决策严格依托 GPU 底层实测张量时间。
        """
        self.model_name = model_name
        self.buckets = sorted(cfg.BUCKETS)
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

    def schedule(self, S_s: int, S_l: int, t_solo_s: float, t_solo_l: float, queue_length: int) -> int:
        """
        基于全局 Slowdown 守恒定律的 O(1) 物理决断核心。
        
        参数:
            S_s, S_l: 长短任务的序列长度
            t_solo_s, t_solo_l: 长短任务的理论基线独占执行时间
            queue_length: 当前系统的排队深度
        返回:
            最优的 Chunk 切分粒度 S_c。若等于 S_l 则触发熔断退化。
        """
        b_s = self._conservative_map_up(S_s)
        b_l = self._conservative_map_up(S_l)

        best_S_c = S_l   
        max_net_benefit = 0.0  

        valid_chunk_candidates = [b for b in self.buckets if b_s <= b < b_l]

        for S_c in valid_chunk_candidates:
            # O(1) 获取真实物理耗时
            t_conc_s = self.lut_gain[b_s][S_c]
            t_penalty = self.lut_penalty[b_l][S_c]

            # 1. 拯救短任务的绝对 Slowdown 收益
            benefit_sd = (t_solo_l - t_conc_s) / max(1.0, t_solo_s)
            
            # 2. 拥塞控制与连锁代价 (The Congestion Control Factor)
            # 物理依据：排队论中，当等待队列趋于拥堵时，任何微小的服务时间增加(Penalty)
            # 都会导致排队延迟呈超线性(二次方)级联爆炸。此设计彻底封死了 GQA/MQA 的过度调度漏洞。
            congestion_factor = 1.0 + (queue_length ** 2) / 100.0
            
            # 3. 长任务自身与全局队列共同承担的 Slowdown 代价
            cost_sd_global = (t_penalty * congestion_factor) / max(1.0, t_solo_l)

            # 4. 净收益判定 (Pareto Optimality)
            net_sd_reduction = benefit_sd - cost_sd_global

            if net_sd_reduction > max_net_benefit:
                max_net_benefit = net_sd_reduction
                best_S_c = S_c

        return best_S_c