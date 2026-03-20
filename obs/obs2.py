import torch
import numpy as np

def benchmark_fairness_final_fixed():
    batch_size = 128 
    d_model = 1024  
    S_long_total = 2048
    S_short = 128
    device = torch.device("cuda:0")
    dtype = torch.float16

    print(f"=== Observation 2: 响应比拯救与【真实物理争用】评估 (Corrected FLOPs) ===")
    print(f"Device: {torch.cuda.get_device_name(0)} | Workload: Full Matrix Multiplication")
    print(f"{'k (Chunks)':<10} | {'Chunk Size':<12} | {'Short Slowdown':<18} | {'Long Slowdown':<18}")
    print("-" * 90)

    # 预分配
    Q_short = torch.randn(batch_size, S_short, d_model, device=device, dtype=dtype)
    K_short_T = torch.randn(batch_size, d_model, S_short, device=device, dtype=dtype)
    out_short = torch.empty(batch_size, S_short, S_short, device=device, dtype=dtype)

    Q_long_full = torch.randn(batch_size, S_long_total, d_model, device=device, dtype=dtype)
    K_long_full_T = torch.randn(batch_size, d_model, S_long_total, device=device, dtype=dtype)
    out_long_full = torch.empty(batch_size, S_long_total, S_long_total, device=device, dtype=dtype)

    def measure_solo_time(Q, K_T, out, iters=50):
        for _ in range(10): torch.bmm(Q, K_T, out=out)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters): torch.bmm(Q, K_T, out=out)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    T_solo_short = measure_solo_time(Q_short, K_short_T, out_short)
    T_solo_long_total = measure_solo_time(Q_long_full, K_long_full_T, out_long_full)
    
    s_short = torch.cuda.Stream(device=device, priority=-1)
    s_long = torch.cuda.Stream(device=device, priority=0)

    for k in [1, 2, 4, 8, 16, 32]:
        chunk_size = S_long_total // k
        
        # 修正：Chunk 必须计算完整列，以保持总计算量一致
        Q_chunk = Q_long_full[:, :chunk_size, :]
        K_long_T = K_long_full_T # 使用完整的 Key
        out_chunk = torch.empty(batch_size, chunk_size, S_long_total, device=device, dtype=dtype)

        if k == 1:
            T_resp_short = T_solo_long_total + T_solo_short
            print(f"{k:<10} | {chunk_size:<12} | {T_resp_short/T_solo_short:<16.2f}x | {1.00:<16.2f}x")
            continue

        iters = 50
        evt_start, evt_end_s, evt_end_l = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        evt_start.record()
        for _ in range(iters):
            with torch.cuda.stream(s_short):
                torch.bmm(Q_short, K_short_T, out=out_short)
                if _ == iters - 1: evt_end_s.record(s_short)
            with torch.cuda.stream(s_long):
                torch.bmm(Q_chunk, K_long_T, out=out_chunk)
                if _ == iters - 1: evt_end_l.record(s_long)
        torch.cuda.synchronize()

        t_true_s = evt_start.elapsed_time(evt_end_s) / iters
        t_true_l_chunk = evt_start.elapsed_time(evt_end_l) / iters
        
        t_solo_l_chunk = measure_solo_time(Q_chunk, K_long_T, out_chunk)

        # 最终指标
        slowdown_short = t_true_s / T_solo_short
        T_resp_long = t_true_l_chunk + (k - 1) * t_solo_l_chunk
        slowdown_long = T_resp_long / T_solo_long_total
        contention = (t_true_s - T_solo_short) / T_solo_short * 100

        print(f"{k:<10} | {chunk_size:<12} | {slowdown_short:<16.2f}x | {slowdown_long:<16.2f}x (Contention: +{contention:.1f}%)")

if __name__ == "__main__":
    benchmark_fairness_final_fixed()