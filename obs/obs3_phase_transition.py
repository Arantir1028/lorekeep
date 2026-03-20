import torch
import time

def benchmark_phase_transition():
    # 物理参数模拟 (以 Llama-2-7B 为例)
    d_model = 4096
    r = 32         # LoRA Rank
    S = 128        # 统一的 Chunk Size (假设已经过了 Wave-Slice 切分)
    device = torch.device("cuda:0")
    dtype = torch.float16

    print("=== Observation 3: 共享基座与异构分支的物理相变 (Phase Transition) ===")
    print(f"{'Batch':<6} | {'Stage':<6} | {'Time (us)':<10} | {'Achieved TFLOPs':<16} | {'Achieved Mem GB/s':<18}")
    print("-" * 65)

    # 预分配 Base Model 的共享权重 (全局唯一)
    W_base = torch.randn(d_model, d_model, device=device, dtype=dtype)

    # 测试不同的并发压力
    batch_sizes = [1, 4, 16, 64, 128]

    for B in batch_sizes:
        # 1. 构造输入数据
        X_base = torch.randn(B * S, d_model, device=device, dtype=dtype) # Base 层输入被打平
        
        # 构造 LoRA 的异构权重 (每个请求拥有自己独立的 W_A 和 W_B)
        # 使用 bmm 模拟最理想的批处理异构 LoRA (真实的 bgmv 算子只会比这个更慢、更吃访存)
        X_lora = torch.randn(B, S, d_model, device=device, dtype=dtype)
        W_A_hetero = torch.randn(B, d_model, r, device=device, dtype=dtype)
        W_B_hetero = torch.randn(B, r, d_model, device=device, dtype=dtype)
        
        out_base = torch.empty(B * S, d_model, device=device, dtype=dtype)
        out_lora_A = torch.empty(B, S, r, device=device, dtype=dtype)
        out_lora_B = torch.empty(B, S, d_model, device=device, dtype=dtype)

        def measure_kernel(func, iters=100):
            for _ in range(10): func() # Warmup
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters): func()
            end.record()
            torch.cuda.synchronize()
            return (start.elapsed_time(end) * 1000) / iters

        # ==========================================
        # 测定 Base Model 阶段 (共享权重 GEMM)
        # ==========================================
        t_base_us = measure_kernel(lambda: torch.mm(X_base, W_base, out=out_base))
        
        # 算力与访存公式推导 (FP16 = 2 Bytes)
        # FLOPs = 2 * (B*S) * d_model * d_model
        flops_base = 2 * (B * S) * d_model * d_model
        # Bytes = 读X + 读W + 写Out = (B*S*d + d*d + B*S*d) * 2
        bytes_base = (B * S * d_model + d_model * d_model + B * S * d_model) * 2
        
        tflops_base = (flops_base / (t_base_us * 1e-6)) / 1e12
        bw_base_gbs = (bytes_base / (t_base_us * 1e-6)) / 1e9

        print(f"{B:<6} | Base   | {t_base_us:<10.2f} | {tflops_base:<16.2f} | {bw_base_gbs:<18.2f}")

        # ==========================================
        # 测定 LoRA 阶段 (异构独立权重 BMM)
        # ==========================================
        def lora_forward():
            torch.bmm(X_lora, W_A_hetero, out=out_lora_A)
            torch.bmm(out_lora_A, W_B_hetero, out=out_lora_B)

        t_lora_us = measure_kernel(lora_forward)
        
        # FLOPs = 2 * (2 * B * S * d_model * r)
        flops_lora = 4 * B * S * d_model * r
        # Bytes = 读X + 读Wa + 写A + 读A + 读Wb + 写Out 
        # 最核心的惩罚：Wa 和 Wb 是不共享的！随 B 线性增长。
        bytes_lora = (B*S*d_model + B*d_model*r + B*S*r) * 2 + (B*S*r + B*r*d_model + B*S*d_model) * 2
        
        tflops_lora = (flops_lora / (t_lora_us * 1e-6)) / 1e12
        bw_lora_gbs = (bytes_lora / (t_lora_us * 1e-6)) / 1e9

        print(f"{B:<6} | LoRA   | {t_lora_us:<10.2f} | {tflops_lora:<16.2f} | {bw_lora_gbs:<18.2f}")
        print("-" * 65)

if __name__ == "__main__":
    benchmark_phase_transition()