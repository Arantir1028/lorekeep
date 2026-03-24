# lorekeep/scheduler/wave_scheduler.py

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg
from scheduler.fairness_engine import FairnessEngine

class WaveScheduler:
    def __init__(self, model_name: str, gamma: float = 2.0):
        self.model_name = model_name
        self.gamma = gamma
        self.buckets = sorted(cfg.BUCKETS)
        self.fairness_engine = FairnessEngine()
        self._load_luts()

    def _load_luts(self):
        paths = cfg.get_lut_paths(self.model_name)
        try:
            with open(paths["raw"], "r") as f:
                raw_data = json.load(f)
                self.t_solo_dict = {int(k): float(v) for k, v in raw_data["T_solo"].items()}
                
            with open(paths["gain"], "r") as f:
                self.lut_gain = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(f).items()}
            with open(paths["penalty"], "r") as f:
                self.lut_penalty = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(f).items()}
        except FileNotFoundError:
            raise Exception(f"Fatal Error: 缺少 {self.model_name} 的底层探针数据。请先运行 Profiler。")

    def _conservative_map_up(self, seq_len: int) -> int:
        for b in self.buckets:
            if b >= seq_len:
                return b
        return self.buckets[-1]

    def schedule_real(self, S_s: int, S_l: int, t_wait_us: float, queue_length: int) -> int:
        """
        核心物理决断引擎。
        输入变更为 queue_length，由系统内部严格推导 rho。
        """
        b_s = self._conservative_map_up(S_s)
        b_l = self._conservative_map_up(S_l)

        t_solo_s = self.t_solo_dict[b_s]
        t_solo_l = self.t_solo_dict[b_l]

        # 1. M/D/1 排队论感知 & 动态 SLA 杠杆
        rho = self.fairness_engine.compute_rho_md1(queue_length)
        w_fairness = self.fairness_engine.compute_weight(t_wait_us, t_solo_s)

        best_S_c = S_l   
        max_net_benefit = 0.0  

        valid_chunk_candidates = [b for b in self.buckets if b_s <= b < b_l]

        for S_c in valid_chunk_candidates:
            t_conc_s = self.lut_gain[b_s][S_c]
            t_penalty = self.lut_penalty[b_l][S_c]

            # 2. 物理边界保护 (The Safeguard)
            # 防止切分过碎导致并发耗时反超独占耗时，产生非法的负向收益
            delta_u = max(0.0, t_solo_l - t_conc_s)
            
            # 3. 拥塞惩罚放大
            cost_global = t_penalty * (1.0 + self.gamma * rho)
            
            # 4. 目标函数决断
            net_benefit = w_fairness * delta_u - cost_global

            if net_benefit > max_net_benefit:
                max_net_benefit = net_benefit
                best_S_c = S_c

        return best_S_c