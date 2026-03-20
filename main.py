# main.py
import time
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams

from engine.vllm_hijacker import inject_wave_slice

def main():
    # 1. 分离完整的 HuggingFace 路径与 Wave-Slice 内部标识名
    hf_model_path = "mistralai/Mistral-7B-v0.1"
    wave_model_name = hf_model_path.split("/")[-1]  # 提取出 "Mistral-7B-v0.1"
    
    # 2. 在引擎启动前，传入纯净的模型名称以加载对应的 LUT 表
    inject_wave_slice(wave_model_name)
    
    # 3. 正常初始化 vLLM 时，仍需使用完整的 HF 路径以下载/加载权重
    engine_args = EngineArgs(
        model=hf_model_path, 
        enable_lora=False,
        max_lora_rank=64,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True, 
        disable_sliding_window=True,
        enforce_eager=True  
    )
    engine = LLMEngine.from_engine_args(engine_args)

    metrics = {}
    
    # 3. 构造 3000 Token 真实超长负载
    prompt_long = "The quick brown fox jumps over the lazy dog. " * 300
    sampling_params_long = SamplingParams(max_tokens=1) 
    
    print("\n[+] 注入长序列背景任务 (Titan)")
    engine.add_request("Long_Titan", prompt_long, sampling_params_long)
    metrics["Long_Titan"] = {"arrival": time.perf_counter(), "ttft": None}

    print("\n--- 启动引擎微观物理步进 ---")
    step_counter = 0
    short_task_id = 0
    
    while engine.has_unfinished_requests():
        step_counter += 1
        
        # 4. 在步进过程中，模拟并发异构任务到达
        if step_counter == 2 or step_counter == 5:
            short_task_id += 1
            req_id = f"Short_Ninja_{short_task_id}"
            prompt_short = "Summarize the key points: " * 5
            sampling_params_short = SamplingParams(max_tokens=1)
            
            print(f" [+] 突发短任务: {req_id} (Tick {step_counter})")
            engine.add_request(req_id, prompt_short, sampling_params_short)
            metrics[req_id] = {"arrival": time.perf_counter(), "ttft": None}
        
        # 触发物理层下发
        request_outputs = engine.step()
        
        current_time = time.perf_counter()
        for output in request_outputs:
            if output.finished:
                req_id = output.request_id
                if metrics[req_id]["ttft"] is None:
                    ttft_ms = (current_time - metrics[req_id]["arrival"]) * 1000 
                    metrics[req_id]["ttft"] = ttft_ms
                    print(f"  >>> [{req_id}] 计算完成 | TTFT: {ttft_ms:.2f} ms")

    print("\n=== Wave-Slice 端到端物理延迟报告 ===")
    print(f"{'Task ID':<20} | {'TTFT (ms)':<15}")
    print("-" * 38)
    for req_id, data in metrics.items():
        ttft_str = f"{data['ttft']:.2f}" if data['ttft'] is not None else "TIMEOUT"
        print(f"{req_id:<20} | {ttft_str:<15}")

if __name__ == "__main__":
    main()