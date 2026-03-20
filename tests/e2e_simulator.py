# lorekeep/tests/e2e_simulator.py

import sys
import os
import random
import json
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg
from scheduler.wave_scheduler import WaveScheduler

# 严格读取真实 GPU 基线
try:
    with open(cfg.RAW_PROFILE_PATH, "r") as f:
        RAW_PROFILE = json.load(f)
        T_SOLO_REAL = {int(k): float(v) for k, v in RAW_PROFILE["T_solo"].items()}
except FileNotFoundError:
    raise Exception("Fatal: 找不到 raw_profile.json！")

class Request:
    def __init__(self, req_id: int, arrival_time: float, is_short: bool):
        self.req_id = req_id
        self.arrival_time = arrival_time
        self.is_short = is_short
        self.seq_len = 64 if is_short else 2048
        # 强制使用真实物理耗时
        self.t_solo = T_SOLO_REAL[self.seq_len]
        self.start_time = -1.0
        self.finish_time = -1.0

    @property
    def slowdown(self) -> float:
        if self.finish_time < 0: return 1.0
        return (self.finish_time - self.arrival_time) / max(1.0, self.t_solo)

class Simulator:
    def __init__(self, scheduler: WaveScheduler, mode="baseline"):
        self.scheduler = scheduler
        self.mode = mode 
        self.current_time = 0.0
        self.short_queue = deque()
        self.long_queue = deque()
        self.completed_requests = []
        
        # O(1) 维护队列的倒数和积分 (替代了虚假的 queue_depth)
        self.sum_inv_tsolo = 0.0

    def add_to_queue(self, req: Request):
        if req.is_short:
            self.short_queue.append(req)
        else:
            self.long_queue.append(req)
        self.sum_inv_tsolo += 1.0 / max(1.0, req.t_solo)

    def pop_from_queue(self, is_short: bool) -> Request:
        req = self.short_queue.popleft() if is_short else self.long_queue.popleft()
        self.sum_inv_tsolo -= 1.0 / max(1.0, req.t_solo)
        # 防止浮点误差累积
        if self.sum_inv_tsolo < 1e-9: self.sum_inv_tsolo = 0.0
        return req

    def run(self, requests: list):
        req_idx = 0
        total_reqs = len(requests)
        
        while req_idx < total_reqs or self.short_queue or self.long_queue:
            # 请求到达
            while req_idx < total_reqs and requests[req_idx].arrival_time <= self.current_time:
                self.add_to_queue(requests[req_idx])
                req_idx += 1

            if not self.short_queue and not self.long_queue:
                if req_idx < total_reqs:
                    self.current_time = requests[req_idx].arrival_time
                continue

            current_batch = []
            if self.short_queue: current_batch.append(self.pop_from_queue(True))
            if self.long_queue: current_batch.append(self.pop_from_queue(False))

            if len(current_batch) == 2:
                req_s, req_l = current_batch[0], current_batch[1]
                req_s.start_time = req_l.start_time = self.current_time

                if self.mode == "baseline":
                    execution_time = req_l.t_solo
                    req_s.finish_time = req_l.finish_time = self.current_time + execution_time
                    self.current_time += execution_time
                
                elif self.mode == "wave-slice":
                    # 大脑执行纯物理决断
                    S_c = self.scheduler.schedule(
                        S_s=req_s.seq_len, S_l=req_l.seq_len,
                        t_solo_s=req_s.t_solo, t_solo_l=req_l.t_solo,
                        sum_inv_tsolo_queue=self.sum_inv_tsolo
                    )
                    
                    if S_c == req_l.seq_len:
                        execution_time = req_l.t_solo
                        req_s.finish_time = req_l.finish_time = self.current_time + execution_time
                        self.current_time += execution_time
                    else:
                        b_s = self.scheduler._conservative_map_up(req_s.seq_len)
                        b_l = self.scheduler._conservative_map_up(req_l.seq_len)
                        t_conc_s = self.scheduler.lut_gain[b_s][S_c]
                        t_penalty = self.scheduler.lut_penalty[b_l][S_c]
                        
                        req_s.finish_time = self.current_time + t_conc_s
                        req_l.finish_time = self.current_time + req_l.t_solo + t_penalty
                        self.current_time += (req_l.t_solo + t_penalty)

                self.completed_requests.extend([req_s, req_l])
            else:
                req = current_batch[0]
                req.start_time = self.current_time
                req.finish_time = self.current_time + req.t_solo
                self.current_time += req.t_solo
                self.completed_requests.append(req)

        return self.completed_requests

def run_evaluation():
    print("=== Wave-Slice E2E Simulator (Zero-Magic-Number Physics Engine) ===")
    
    scheduler = WaveScheduler()
    t_solo_long = T_SOLO_REAL[2048]
    max_lambda = 1.0 / t_solo_long 
    
    # 我们测试逼近系统死亡红线的负载
    load_factors = [0.10, 0.50, 0.80, 0.95, 0.99]
    lambdas = [f * max_lambda for f in load_factors]
    num_reqs = 1000
    
    print(f"检测到 A100 长任务真实基线: {t_solo_long:.2f} us")
    print(f"{'系统负载 (rho)':<15} | {'Baseline P99 Slowdown':<30} | {'Wave-Slice P99 Slowdown':<30}")
    print("-" * 80)
    
    for load_factor, lam in zip(load_factors, lambdas):
        random.seed(42)
        # 生成请求时传递真实的 lambda
        requests_base = []
        t = 0.0
        for i in range(num_reqs):
            t += random.expovariate(lam)
            requests_base.append(Request(i, t, random.random() < 0.8))
            
        # 深度拷贝一份给 wave-slice，确保两边面对的排队宇宙绝对一致
        import copy
        requests_wave = copy.deepcopy(requests_base)
        
        sim_base = Simulator(scheduler, mode="baseline")
        completed_base = sim_base.run(requests_base)
        slowdowns_s_base = [r.slowdown for r in completed_base if r.is_short]
        p99_base = np.percentile(slowdowns_s_base, 99) if slowdowns_s_base else 1.0
        
        sim_wave = Simulator(scheduler, mode="wave-slice")
        completed_wave = sim_wave.run(requests_wave)
        slowdowns_s_wave = [r.slowdown for r in completed_wave if r.is_short]
        p99_wave = np.percentile(slowdowns_s_wave, 99) if slowdowns_s_wave else 1.0
        
        print(f"{load_factor*100:>5.0f}% ({lam:.6f}) | {p99_base:<30.2f} | {p99_wave:<30.2f}")

if __name__ == "__main__":
    run_evaluation()