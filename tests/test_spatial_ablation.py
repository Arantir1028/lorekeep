# tests/test_spatial_ablation.py

import torch
import time
import sys
import os

# 将项目根目录加入路径，以便导入 engine 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.lora_unbinder import WaveLoRAUnbinder

def run_spatial_ablation(d_model=4096, r=64, S_l=1024, S_s=1, num_warmup=10, num_iters=50):
    print("\n" + "="*70)
    print(f"🔬 启动物理消融实验: 空域重叠 (Spatial) vs 时域串行 (Temporal)")
    print(f"   负载参数: d_model={d_model}, rank={r}, 长任务={S_l}, 短任务={S_s}")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠️ 跳过: 当前环境无 CUDA，无法验证多流 LoRA 解绑有效性。")
        return
    
    # 1. 物理环境初始化
    unbinder = WaveLoRAUnbinder(d_model, r, device)
    
    # 模拟 Base Model 输出的隐藏状态 (Hidden States)
    X_base_out = torch.randn(S_l + S_s, d_model, device=device, dtype=torch.float16)
    
    # 模拟独立的 LoRA 权重
    lora_A_short = torch.randn(d_model, r, device=device, dtype=torch.float16)
    lora_B_short = torch.randn(r, d_model, device=device, dtype=torch.float16)
    lora_A_long = torch.randn(d_model, r, device=device, dtype=torch.float16)
    lora_B_long = torch.randn(r, d_model, device=device, dtype=torch.float16)

    # CUDA 事件探针
    start_event = torch.cuda.Event(enable_timing=True)
    end_short_event = torch.cuda.Event(enable_timing=True)
    end_long_event = torch.cuda.Event(enable_timing=True)

    def benchmark_temporal_serial():
        """
        基线：时域串行 (Temporal Only / vLLM Default)
        在同一个 Default Stream 中，先执行长任务，再执行短任务。短任务被迫排队。
        """
        torch.cuda.synchronize()
        start_event.record()
        
        # 长任务计算
        X_long = X_base_out[S_s:, :]
        proj_long = X_long @ lora_A_long @ lora_B_long
        end_long_event.record()
        
        # 短任务被迫等待长任务完成
        X_short = X_base_out[:S_s, :]
        proj_short = X_short @ lora_A_short @ lora_B_short
        end_short_event.record()
        
        torch.cuda.synchronize()
        t_short = start_event.elapsed_time(end_short_event) * 1000 # 转为微秒 us
        t_total = start_event.elapsed_time(end_long_event) * 1000
        return t_short, t_total

    def benchmark_spatio_temporal():
        """
        Ours：时空协同 (Spatio-Temporal / Wave-Slice Phase II)
        利用多流在空域上实现物理重叠，短任务瞬间逃逸。
        """
        torch.cuda.synchronize()
        start_event.record()
        
        # 调用我们写好的多流解绑器
        unbinder.execute_and_escape(
            X_base_out=X_base_out,
            S_s=S_s,
            S_l=S_l,
            lora_A_short=lora_A_short,
            lora_B_short=lora_B_short,
            lora_A_long=lora_A_long,
            lora_B_long=lora_B_long,
            evt_short_done=end_short_event
        )
        end_long_event.record(unbinder.stream_long)
        
        torch.cuda.synchronize()
        t_short = start_event.elapsed_time(end_short_event) * 1000 # 微秒
        t_total = start_event.elapsed_time(end_long_event) * 1000
        return t_short, t_total

    # 2. Warmup 预热 (消除 CUDA Context 初始化抖动)
    for _ in range(num_warmup):
        benchmark_temporal_serial()
        benchmark_spatio_temporal()

    # 3. 正式压测
    t_short_temporal_list, t_short_spatial_list = [], []
    
    for _ in range(num_iters):
        t_s_temp, _ = benchmark_temporal_serial()
        t_s_spat, _ = benchmark_spatio_temporal()
        t_short_temporal_list.append(t_s_temp)
        t_short_spatial_list.append(t_s_spat)

    # 4. 数据汇总
    avg_temp_s = sum(t_short_temporal_list) / num_iters
    avg_spat_s = sum(t_short_spatial_list) / num_iters
    
    print(f"\n📊 [短任务 SLA 响应延迟 (Microseconds)]")
    print(f"  ❌ 基线 (纯时域串行) : {avg_temp_s:.2f} us")
    print(f"  ✅ Ours (时空多流并发): {avg_spat_s:.2f} us")
    
    if avg_temp_s > avg_spat_s:
        speedup = avg_temp_s / avg_spat_s
        print(f"\n🚀 结论: 空域复用成功！短任务逃逸速度提升了 {speedup:.2f} 倍。")
        print(f"   (这证明了长任务切块后确实产生了 SM 气泡，多流重叠物理有效)")
    else:
        print(f"\n⚠️ 警告: 空域复用未见明显收益。可能原因是 GPU 算力依然满载，或切片粒度需调整。")

if __name__ == "__main__":
    run_spatial_ablation()
