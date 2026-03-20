# lorekeep/tests/e2e_simulator.py
import sys, os, random, json, copy
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg
from scheduler.wave_scheduler import WaveScheduler

# SOTA 的静态分块大小 (vLLM 默认配置通常为 256 或 512)
STATIC_CHUNK_SIZE = 512

class Request:
    def __init__(self, req_id: int, arrival_time: float, is_short: bool, t_solo_dict: dict):
        self.req_id = req_id
        self.arrival_time = arrival_time
        self.is_short = is_short
        self.seq_len = 64 if is_short else 2048
        self.t_solo = t_solo_dict[self.seq_len]
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
        self.sum_inv_tsolo = 0.0

    def add_to_queue(self, req: Request):
        if req.is_short: self.short_queue.append(req)
        else: self.long_queue.append(req)
        self.sum_inv_tsolo += 1.0 / max(1.0, req.t_solo)

    def pop_from_queue(self, is_short: bool) -> Request:
        req = self.short_queue.popleft() if is_short else self.long_queue.popleft()
        self.sum_inv_tsolo -= 1.0 / max(1.0, req.t_solo)
        if self.sum_inv_tsolo < 1e-9: self.sum_inv_tsolo = 0.0
        return req

    def _get_static_penalty(self, seq_len: int) -> float:
        """获取 SOTA 静态切分带来的不可避免的物理惩罚"""
        if seq_len <= STATIC_CHUNK_SIZE: return 0.0
        b_l = self.scheduler._conservative_map_up(seq_len)
        b_c = self.scheduler._conservative_map_up(STATIC_CHUNK_SIZE)
        return self.scheduler.lut_penalty[b_l][b_c]

    def run(self, requests: list):
        req_idx, total_reqs = 0, len(requests)
        while req_idx < total_reqs or self.short_queue or self.long_queue:
            while req_idx < total_reqs and requests[req_idx].arrival_time <= self.current_time:
                self.add_to_queue(requests[req_idx])
                req_idx += 1

            if not self.short_queue and not self.long_queue:
                if req_idx < total_reqs: self.current_time = requests[req_idx].arrival_time
                continue

            current_batch = []
            if self.short_queue: current_batch.append(self.pop_from_queue(True))
            if self.long_queue: current_batch.append(self.pop_from_queue(False))

            if len(current_batch) == 2:
                req_s, req_l = current_batch[0], current_batch[1]
                req_s.start_time = req_l.start_time = self.current_time

                if self.mode == "baseline":
                    exec_time = req_l.t_solo
                    req_s.finish_time = req_l.finish_time = self.current_time + exec_time
                    self.current_time += exec_time
                    
                elif self.mode == "vllm-static":
                    # SOTA 静态切分：无脑将长任务切分为 512
                    b_s = self.scheduler._conservative_map_up(req_s.seq_len)
                    b_c = self.scheduler._conservative_map_up(STATIC_CHUNK_SIZE)
                    t_conc_s = self.scheduler.lut_gain[b_s][b_c]
                    t_penalty = self._get_static_penalty(req_l.seq_len)
                    
                    req_s.finish_time = self.current_time + t_conc_s
                    req_l.finish_time = self.current_time + req_l.t_solo + t_penalty
                    self.current_time += (req_l.t_solo + t_penalty)

                elif self.mode == "wave-slice":
                    S_c = self.scheduler.schedule(req_s.seq_len, req_l.seq_len, req_s.t_solo, req_l.t_solo, self.sum_inv_tsolo)
                    if S_c == req_l.seq_len: # 动态熔断，回归 Baseline
                        exec_time = req_l.t_solo
                        req_s.finish_time = req_l.finish_time = self.current_time + exec_time
                        self.current_time += exec_time
                    else: # 动态最优切分
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
                # 即使是独占执行，SOTA 依然会强制切分长任务，付出切分代价
                penalty = self._get_static_penalty(req.seq_len) if self.mode == "vllm-static" else 0.0
                req.finish_time = self.current_time + req.t_solo + penalty
                self.current_time += (req.t_solo + penalty)
                self.completed_requests.append(req)

        return self.completed_requests

def run_evaluation_for_model(model_name: str):
    paths = cfg.get_lut_paths(model_name)
    try:
        with open(paths["raw"], "r") as f:
            raw_profile = json.load(f)
            t_solo_real = {int(k): float(v) for k, v in raw_profile["T_solo"].items()}
    except FileNotFoundError:
        print(f"⚠️ 跳过 {model_name}: 缺少底层探针数据。")
        return

    scheduler = WaveScheduler(model_name)
    t_solo_long = t_solo_real[2048]
    max_lambda = 1.0 / t_solo_long 
    
    load_factors = [0.10, 0.50, 0.80, 0.95, 0.99]
    lambdas = [f * max_lambda for f in load_factors]
    num_reqs = 1000
    
    print(f"\n✅ 模型: {model_name} | 架构: {cfg.SUPPORTED_MODELS[model_name]['attn_type']} | 长任务基线: {t_solo_long:.2f} us")
    print(f"{'负载 (rho)':<12} | {'Baseline P99':<16} | {'vLLM SOTA P99':<16} | {'Wave-Slice P99':<16}")
    print("-" * 72)
    
    for load_factor, lam in zip(load_factors, lambdas):
        random.seed(42)
        requests_base = []
        t = 0.0
        for i in range(num_reqs):
            t += random.expovariate(lam)
            requests_base.append(Request(i, t, random.random() < 0.8, t_solo_real))
            
        requests_vllm = copy.deepcopy(requests_base)
        requests_wave = copy.deepcopy(requests_base)
        
        # Baseline
        sim_base = Simulator(scheduler, mode="baseline")
        completed_base = sim_base.run(requests_base)
        p99_base = np.percentile([r.slowdown for r in completed_base if r.is_short], 99)
        
        # vLLM Static SOTA
        sim_vllm = Simulator(scheduler, mode="vllm-static")
        completed_vllm = sim_vllm.run(requests_vllm)
        p99_vllm = np.percentile([r.slowdown for r in completed_vllm if r.is_short], 99)
        
        # Wave-Slice
        sim_wave = Simulator(scheduler, mode="wave-slice")
        completed_wave = sim_wave.run(requests_wave)
        p99_wave = np.percentile([r.slowdown for r in completed_wave if r.is_short], 99)
        
        print(f"{load_factor*100:>3.0f}% ({lam:.6f})| {p99_base:<16.2f} | {p99_vllm:<16.2f} | {p99_wave:<16.2f}")

if __name__ == "__main__":
    print("=== Wave-Slice E2E Simulator (vs vLLM SOTA Evaluation) ===")
    for model in cfg.SUPPORTED_MODELS.keys():
        run_evaluation_for_model(model)