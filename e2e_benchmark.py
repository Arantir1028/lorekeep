# lorekeep/e2e_benchmark.py
import time
import random
import numpy as np
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
from engine.vllm_hijacker import inject_wave_slice

def run_simulation(engine: LLMEngine, short_arrival_rate: float, mode_name: str, num_short_requests: int = 50):
    print(f"\n{'='*70}")
    print(f"🔥 启动极限抗压测试: {mode_name} | 短任务并发率 (λ): {short_arrival_rate} Req/s")
    print(f"{'='*70}")

    requests = []
    
    # 1. T=0 瞬间注入 3 个超级背景长任务 (制造极端的 Running 队列拥塞)
    for i in range(3):
        requests.append({
            "req_id": f"Long_Titan_Bg_{i}",
            "prompt": "The capital of the world is a complex topic. " * 3000, # 约 24000 Tokens
            "is_short": False,
            "arrival_time": 0.0, 
            "injected": False
        })

    # 2. 生成高频短任务流 (泊松到达)
    random.seed(42)
    current_arrival_time = 0.5 # 让长任务先跑半秒钟，彻底占领引擎
    
    for i in range(num_short_requests):
        inter_arrival = random.expovariate(short_arrival_rate)
        current_arrival_time += inter_arrival
        
        requests.append({
            "req_id": f"Short_Ninja_{i}",
            "prompt": "Translate this to French: The weather is very nice today. " * 3,
            "is_short": True,
            "arrival_time": current_arrival_time,
            "injected": False
        })

    metrics = {}
    start_time = time.perf_counter()
    step_counter = 0

    while True:
        elapsed_time = time.perf_counter() - start_time
        
        # 注入到达的请求
        for req in requests:
            if not req["injected"] and req["arrival_time"] <= elapsed_time:
                engine.add_request(
                    req["req_id"], 
                    req["prompt"], 
                    SamplingParams(max_tokens=1, temperature=0.0)
                )
                req["injected"] = True
                metrics[req["req_id"]] = {
                    "is_short": req["is_short"],
                    "arrival_real_time": time.perf_counter(),
                    "ttft": None
                }

        # 退出条件：全部注入且全部短任务算完 (为节省评测时间，不强制等长任务算完)
        all_short_injected = all(r["injected"] for r in requests if r["is_short"])
        all_short_finished = all(metrics.get(r["req_id"], {}).get("ttft") is not None for r in requests if r["is_short"])
        if all_short_injected and all_short_finished:
            # 强行终止未完成的长任务，重置引擎状态供下一轮使用
            for req in requests:
                if not req["is_short"] and metrics.get(req["req_id"], {}).get("ttft") is None:
                    engine.abort_request(req["req_id"])
            break
            
        if engine.has_unfinished_requests():
            step_counter += 1
            request_outputs = engine.step()
            current_time = time.perf_counter()
            
            for output in request_outputs:
                req_id = output.request_id
                if metrics[req_id]["ttft"] is None and len(output.outputs[0].token_ids) > 0:
                    ttft_ms = (current_time - metrics[req_id]["arrival_real_time"]) * 1000 
                    metrics[req_id]["ttft"] = ttft_ms

    # 3. 统计 P99 数据 (仅统计短任务，看 SLA 是否被保护)
    short_ttfts = [d["ttft"] for d in metrics.values() if d["is_short"] and d["ttft"] is not None]
    p99_short = np.percentile(short_ttfts, 99) if short_ttfts else 0
    mean_short = np.mean(short_ttfts) if short_ttfts else 0
    
    print(f"\n=== 仿真结果 ({mode_name} | λ={short_arrival_rate}) ===")
    print(f"短任务 P99 TTFT: {p99_short:.2f} ms (Mean: {mean_short:.2f} ms)")
    
    return p99_short

if __name__ == "__main__":
    hf_model_path = "mistralai/Mistral-7B-v0.1"
    
    USE_WAVE_SLICE = False # <--- 先用 True 跑，然后改 False 重开终端跑
    
    if USE_WAVE_SLICE:
        wave_model_name = hf_model_path.split("/")[-1]
        inject_wave_slice(wave_model_name)
        mode_name = "Wave-Slice 劫持模式"
    else:
        import vllm.core.scheduler as scheduler_module
        if hasattr(scheduler_module.Scheduler, "_wave_schedule_hook"):
            raise RuntimeError("检测到劫持污染！请彻底关闭当前终端，新开一个终端来运行 Baseline。")
        mode_name = "vLLM 原生 Baseline"

    engine_args = EngineArgs(
        model=hf_model_path, 
        enable_lora=False,
        max_num_batched_tokens=2048, 
        enable_chunked_prefill=True, 
        disable_sliding_window=True,
        enforce_eager=True  
    )
    
    print(f"\n💡 [System] 正在初始化全局 vLLM 引擎...")
    engine = LLMEngine.from_engine_args(engine_args)
    
    # 将压力提升到能让引擎排队的水平
    load_levels = [5.0, 10.0, 20.0, 30.0] 
    
    results = {}
    for rate in load_levels:
        p99 = run_simulation(engine, rate, mode_name, num_short_requests=40)
        results[rate] = p99
        
    print(f"\n📊 最终汇总帕累托数据 (模式: {mode_name})")
    print(f"{'Short Arrival Rate (λ)':<25} | {'Short Task P99 TTFT (ms)':<25}")
    print("-" * 55)
    for rate, p99 in results.items():
        print(f"{rate:<25.1f} | {p99:<25.2f}")