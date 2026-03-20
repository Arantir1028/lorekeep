# lorekeep/scheduler/wave_scheduler.py

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg

class WaveScheduler:
    def __init__(self):
        """
        初始化纯物理引擎。
        摒弃所有魔法数字 (gamma, max_queue_depth, W_max 等全部删除)。
        """
        self.buckets = sorted(cfg.BUCKETS)
        self._load_luts()

    def _load_luts(self):
        try:
            with open(cfg.LUT_GAIN_PATH, "r") as f:
                self.lut_gain = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(f).items()}
            with open(cfg.LUT_PENALTY_PATH, "r") as f:
                self.lut_penalty = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(f).items()}
            print("✅ Physical Scheduler: Hardware LUTs strictly loaded.")
        except FileNotFoundError:
            raise Exception("Fatal: 找不到 LUT 表，必须基于真实硬件探针数据运行！")

    def _conservative_map_up(self, seq_len: int) -> int:
        for b in self.buckets:
            if b >= seq_len:
                return b
        return self.buckets[-1]

    def schedule(self, S_s: int, S_l: int, t_solo_s: float, t_solo_l: float, sum_inv_tsolo_queue: float) -> int:
        """
        基于全局 Slowdown 守恒定律的 O(1) 物理决断。
        """
        b_s = self._conservative_map_up(S_s)
        b_l = self._conservative_map_up(S_l)

        best_S_c = S_l   # 默认退化 (不切分)
        max_net_benefit = 0.0  # 净收益必须严格大于 0 才允许切分

        valid_chunk_candidates = [b for b in self.buckets if b_s <= b < b_l]

        for S_c in valid_chunk_candidates:
            # 提取真实物理数据
            t_conc_s = self.lut_gain[b_s][S_c]
            t_penalty = self.lut_penalty[b_l][S_c]

            # 1. 拯救短任务的 Slowdown 收益
            benefit_sd = (t_solo_l - t_conc_s) / max(1.0, t_solo_s)
            
            # 2. 长任务自身受损的 Slowdown 代价
            cost_sd_long = t_penalty / max(1.0, t_solo_l)
            
            # 3. 全局队列受损的连锁 Slowdown 代价
            cost_sd_queue = t_penalty * sum_inv_tsolo_queue

            # 4. 物理净收益
            net_sd_reduction = benefit_sd - (cost_sd_long + cost_sd_queue)

            if net_sd_reduction > max_net_benefit:
                max_net_benefit = net_sd_reduction
                best_S_c = S_c

        return best_S_c

# 单元测试与验证
if __name__ == "__main__":
    scheduler = WaveScheduler(gamma=0.5, max_queue_depth=100)
    
    # 模拟场景：短任务 45 Token，长任务 1800 Token
    S_short, S_long = 45, 1800
    T_solo_short = 50.0  # 理想耗时 50us
    
    print(f"\n=== Wave-Slice O(1) 调度决断推演 ===")
    print(f"请求特征: 短任务 S_s={S_short}, 长任务 S_l={S_long}")
    
    # 场景 A: 队列极度空闲，短任务刚来 (不急)
    sc_A = scheduler.schedule(S_short, S_long, t_wait_s_us=5.0, t_solo_s_us=T_solo_short, current_queue_depth=5)
    print(f"场景 A (轻负载, 短任务不急): 决断 S_c = {sc_A} (退化为 Grouped GEMM)")
    
    # 场景 B: 队列极度空闲，但短任务被卡了很久 (触发 SLA 拯救)
    sc_B = scheduler.schedule(S_short, S_long, t_wait_s_us=300.0, t_solo_s_us=T_solo_short, current_queue_depth=5)
    print(f"场景 B (轻负载, 短任务极度饥饿): 决断 S_c = {sc_B} (执行激进切分)")
    
    # 场景 C: 队列极度拥挤，短任务被卡了很久 (触发 Elastic Bypass 保吞吐)
    sc_C = scheduler.schedule(S_short, S_long, t_wait_s_us=300.0, t_solo_s_us=T_solo_short, current_queue_depth=99)
    print(f"场景 C (系统爆满, 短任务饥饿): 决断 S_c = {sc_C} (触发 Elastic Bypass, 拒绝切分保吞吐)")