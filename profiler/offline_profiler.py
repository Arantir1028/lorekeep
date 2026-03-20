# lorekeep/profiler/offline_profiler.py

import torch
import torch.nn.functional as F
import json
import itertools
from tqdm import tqdm
import sys
import os

# 引入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg

def measure_solo_time(Q, K, V, iters=50):
    """测量极致纯净的独占执行时间 T_solo"""
    # 预热
    for _ in range(10): 
        F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000) / iters  # 转换为微秒 (us)

def measure_concurrent_time(S_s, S_c, device, dtype, iters=30):
    """测量存在物理争用的多流并发时间 T_conc (使用异步拓扑屏障，摒弃过度工程的 CUDA Graph)"""
    Q_s = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, S_s, cfg.HEAD_DIM, device=device, dtype=dtype)
    K_s = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, S_s, cfg.HEAD_DIM, device=device, dtype=dtype)
    V_s = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, S_s, cfg.HEAD_DIM, device=device, dtype=dtype)
    
    Q_c = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, S_c, cfg.HEAD_DIM, device=device, dtype=dtype)
    K_c = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, S_c, cfg.HEAD_DIM, device=device, dtype=dtype)
    V_c = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, S_c, cfg.HEAD_DIM, device=device, dtype=dtype)

    s_main = torch.cuda.current_stream()
    s_short = torch.cuda.Stream(device=device, priority=-1)
    s_long = torch.cuda.Stream(device=device, priority=0)

    # 1. 深度预热 (确保 SDPA 后端启发式算法完全收敛，分配好所有必要的 Workspace)
    for _ in range(5):
        s_short.wait_stream(s_main)
        s_long.wait_stream(s_main)
        with torch.cuda.stream(s_short):
            F.scaled_dot_product_attention(Q_s, K_s, V_s, is_causal=True)
        with torch.cuda.stream(s_long):
            F.scaled_dot_product_attention(Q_c, K_c, V_c, is_causal=True)
        s_main.wait_stream(s_short)
        s_main.wait_stream(s_long)
    torch.cuda.synchronize()

    # 2. 物理测量 (强制 1:1 并发对齐)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    
    for _ in range(iters):
        # Fork 屏障：确保本轮的长短任务同时起步
        s_short.wait_stream(s_main)
        s_long.wait_stream(s_main)
        
        with torch.cuda.stream(s_short):
            F.scaled_dot_product_attention(Q_s, K_s, V_s, is_causal=True)
        with torch.cuda.stream(s_long):
            F.scaled_dot_product_attention(Q_c, K_c, V_c, is_causal=True)
            
        # Join 屏障：确保本轮双流计算结束，才允许主流进入下一轮 Fork。
        # 注意：这里的 wait_stream 是异步的，不会阻塞 CPU，能将 GPU 压榨到极限。
        s_main.wait_stream(s_short)
        s_main.wait_stream(s_long)
        
    end.record()
    torch.cuda.synchronize()
    
    return (start.elapsed_time(end) * 1000) / iters

def measure_read_amp_sum(S_l, S_c, device, dtype, iters=30):
    """
    【修正版】测量切分导致的历史 KV Cache 读放大总耗时。
    采用连续时序模拟（Continuous Sequential Simulation），完美保留 L2 Cache 局部性与 Thrashing 现象。
    """
    if S_c >= S_l:
        return 0.0
    
    k = S_l // S_c
    if S_l % S_c != 0:
        k += 1 # 包含残差块
        
    # 全局 KV Cache (模拟 Base 阶段累积的 Cache)
    K_full = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, S_l, cfg.HEAD_DIM, device=device, dtype=dtype)
    V_full = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, S_l, cfg.HEAD_DIM, device=device, dtype=dtype)
    
    # 提前准备好每个 Chunk 的 Query，避免在测速循环内分配内存
    q_chunks = []
    history_lens = []
    for i in range(1, k):
        current_seq_len = min(S_c, S_l - i * S_c)
        history_len = i * S_c + current_seq_len
        q_chunks.append(torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, current_seq_len, cfg.HEAD_DIM, device=device, dtype=dtype))
        history_lens.append(history_len)

    s_main = torch.cuda.current_stream()
    
    # 1. 预热 (让 GPU 预分配好内部 Workspace)
    for _ in range(3):
        for i in range(k - 1):
            F.scaled_dot_product_attention(q_chunks[i], K_full[:, :, :history_lens[i], :], V_full[:, :, :history_lens[i], :], is_causal=True)
    torch.cuda.synchronize()

    # 2. 连续物理执行测速 (核心：不要在内部 synchronize，让 Chunk 像流水线一样连续流过 L2 Cache)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    
    for _ in range(iters):
        for i in range(k - 1):
            # 严格模拟真实推理时序：Chunk i 算完立刻算 Chunk i+1，L2 会自然缓存 K_full 和 V_full
            F.scaled_dot_product_attention(q_chunks[i], K_full[:, :, :history_lens[i], :], V_full[:, :, :history_lens[i], :], is_causal=True)
            
    end.record()
    torch.cuda.synchronize()
    
    # 减去理想状态下的纯计算时间 (因为我们要的仅仅是 "放大带来的额外惩罚")
    total_continuous_time_us = (start.elapsed_time(end) * 1000) / iters
    
    # 理论上这 k-1 个 Chunk 的纯计算总时间近似等于 (S_l - S_c) 长度的 Solo 时间
    # 这里通过提取 LUT 中的基准时间进行惩罚扣减
    # 注意：在真实的 LUT_Generator 中，会直接用连续执行总时间与 T_solo 组合做减法。
    # 这里探针直接返回串行执行这 k-1 个 chunk 的总微秒数。
    return total_continuous_time_us

def run_profiler():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"=== Starting Wave-Slice Hardware Profiler on {torch.cuda.get_device_name(device)} ===")
    
    profile_data = {
        "T_solo": {},       # {S: time}
        "T_conc": {},       # {S_s: {S_c: time}}
        "T_read_amp": {}    # {S_l: {S_c: time}}
    }

    # 1. 采集 T_solo
    print("\n[1/3] Profiling T_solo (Ideal Baseline)...")
    for b in tqdm(cfg.BUCKETS):
        Q = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, b, cfg.HEAD_DIM, device=device, dtype=dtype)
        K = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, b, cfg.HEAD_DIM, device=device, dtype=dtype)
        V = torch.randn(cfg.BATCH_SIZE, cfg.NUM_HEADS, b, cfg.HEAD_DIM, device=device, dtype=dtype)
        profile_data["T_solo"][b] = measure_solo_time(Q, K, V)

    # 2. 采集 T_conc (物理争用)
    print("\n[2/3] Profiling T_conc (Multi-Stream Contention)...")
    for s_s in tqdm(cfg.BUCKETS):
        profile_data["T_conc"][s_s] = {}
        for s_c in cfg.BUCKETS:
            # 物理限制：Chunk 必须大于等于短任务，否则切分无意义
            if s_c >= s_s:
                profile_data["T_conc"][s_s][s_c] = measure_concurrent_time(s_s, s_c, device, dtype)

    # 3. 采集 T_read_amp (读放大惩罚)
    print("\n[3/3] Profiling T_read_amp (KV Cache Read Amplification)...")
    for s_l in tqdm(cfg.BUCKETS):
        profile_data["T_read_amp"][s_l] = {}
        for s_c in cfg.BUCKETS:
            if s_c < s_l:
                profile_data["T_read_amp"][s_l][s_c] = measure_read_amp_sum(s_l, s_c, device, dtype)
            else:
                profile_data["T_read_amp"][s_l][s_c] = 0.0

    # 写入 JSON
    with open(cfg.RAW_PROFILE_PATH, "w") as f:
        json.dump(profile_data, f, indent=4)
    print(f"\n✅ Raw profile data saved to {cfg.RAW_PROFILE_PATH}")

if __name__ == "__main__":
    run_profiler()