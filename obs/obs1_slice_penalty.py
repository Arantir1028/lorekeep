import torch
import torch.nn.functional as F
import numpy as np

def benchmark_slicing_cost():
    # =================================================================
    # 1. 物理参数模拟 (以 Llama-2-7B 单层 Attention 为例)
    # =================================================================
    S_total = 2048          # 长任务总序列长度
    batch_size = 1
    num_heads = 32
    head_dim = 128
    device = torch.device("cuda:0")
    dtype = torch.float16

    print(f"=== Observation 1: The Cost of Slicing (S={S_total}) ===")
    print(f"{'Chunks (k)':<12} | {'Chunk Size':<12} | {'Total Time (us)':<16} | {'Slowdown Penalty':<16}")
    print("-" * 65)

    # 预先分配好全局的 Q, K, V (模拟 Base Model 阶段生成的完整序列特征)
    Q_full = torch.randn(batch_size, num_heads, S_total, head_dim, device=device, dtype=dtype)
    K_full = torch.randn(batch_size, num_heads, S_total, head_dim, device=device, dtype=dtype)
    V_full = torch.randn(batch_size, num_heads, S_total, head_dim, device=device, dtype=dtype)

    # 候选的切分次数 k (1 代表不切分，即 SOTA Baseline)
    k_candidates = [1, 2, 4, 8, 16, 32, 64]
    
    baseline_time = None

    for k in k_candidates:
        chunk_size = S_total // k
        
        # 验证能否整除
        if S_total % k != 0:
            continue

        def simulate_chunked_prefill():
            # 模拟 vLLM Chunked Prefill 的物理过程：
            # 切分为 k 个块，第 i 个块的 Query 只有 chunk_size 长，
            # 但它的 Key 和 Value 必须包含历史所有的 Token (长度为 (i+1) * chunk_size)
            for i in range(k):
                q_chunk = Q_full[:, :, i*chunk_size : (i+1)*chunk_size, :]
                k_cache = K_full[:, :, : (i+1)*chunk_size, :]
                v_cache = V_full[:, :, : (i+1)*chunk_size, :]
                
                # 底层自动调用 FlashAttention / Memory-efficient Attention
                out = F.scaled_dot_product_attention(q_chunk, k_cache, v_cache, is_causal=True)
            return out

        # 预热
        for _ in range(10):
            simulate_chunked_prefill()
        torch.cuda.synchronize()

        # 高精度测速
        num_iters = 100
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            start_events[i].record()
            simulate_chunked_prefill()
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
        avg_time_us = np.mean(times)

        # 记录基线时间 (k=1)
        if k == 1:
            baseline_time = avg_time_us
            slowdown = 1.0
        else:
            slowdown = avg_time_us / baseline_time

        print(f"{k:<12} | {chunk_size:<12} | {avg_time_us:<16.2f} | {slowdown:<14.2f}x")

if __name__ == "__main__":
    benchmark_slicing_cost()