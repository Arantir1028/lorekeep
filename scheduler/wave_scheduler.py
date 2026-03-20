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
            # 新增：直接读取离线探针的真实理想执行时间 T_solo
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

    def schedule_real(self, S_s: int, S_l: int, t_wait_us: float, rho: float) -> int:
        """真实业务侧调用的 O(1) 决断接口"""
        b_s = self._conservative_map_up(S_s)
        b_l = self._conservative_map_up(S_l)

        # 提取真实的基线纯计算时间
        t_solo_s = self.t_solo_dict[b_s]
        t_solo_l = self.t_solo_dict[b_l]

        best_S_c = S_l   
        max_net_benefit = 0.0  

        valid_chunk_candidates = [b for b in self.buckets if b_s <= b < b_l]
        
        w_fairness = self.fairness_engine.compute_weight(t_wait_us, t_solo_s)

        for S_c in valid_chunk_candidates:
            t_conc_s = self.lut_gain[b_s][S_c]
            t_penalty = self.lut_penalty[b_l][S_c]

            delta_u = t_solo_l - t_conc_s
            cost_global = t_penalty * (1.0 + self.gamma * rho)
            net_benefit = w_fairness * delta_u - cost_global

            if net_benefit > max_net_benefit:
                max_net_benefit = net_benefit
                best_S_c = S_c

        return best_S_c