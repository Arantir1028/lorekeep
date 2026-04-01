# main.py
import time

from engine.runtime_bootstrap import bootstrap_vllm_runtime
from engine.vllm_hijacker import inject_wave_slice

bootstrap_vllm_runtime()

try:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine
    from vllm.sampling_params import SamplingParams
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "vllm is not installed. Install it first, e.g. `pip install vllm`, "
        "then re-run main.py."
    ) from exc

def setup_engine(model_path="mistralai/Mistral-7B-v0.1", enable_lora: bool = False):
    wave_model_name = model_path.split("/")[-1] 
    
    # 物理劫持注入
    inject_wave_slice(wave_model_name)
    
    engine_args = EngineArgs(
        model=model_path, 
        enable_lora=enable_lora,
        max_lora_rank=32,
        max_num_batched_tokens=2048, # 维持较小预算，逼迫引擎触发多帧切分
        enable_chunked_prefill=True, 
        disable_sliding_window=True,
        enforce_eager=True  
    )
    return LLMEngine.from_engine_args(engine_args)

# =====================================================================
# [历史验证资产保留]：微观物理延迟与 TTFT 倒挂验证 (Tick-by-Tick)
# =====================================================================
def run_latency_benchmark(engine):
    print("\n" + "="*50)
    print("🚀 启动模式: 微观物理延迟与 TTFT 倒挂测试")
    print("="*50)
    metrics = {}
    
    # Keep within common 4k context windows across vLLM versions.
    prompt_long = "The quick brown fox jumps over the lazy dog. " * 200
    sampling_params_long = SamplingParams(max_tokens=16)
    
    print("[+] 注入长序列背景任务 (Titan)")
    engine.add_request("Long_Titan", prompt_long, sampling_params_long)
    metrics["Long_Titan"] = {"arrival": time.perf_counter(), "ttft": None}

    step_counter = 0
    short_task_id = 0
    
    while engine.has_unfinished_requests():
        step_counter += 1
        if step_counter == 1 or step_counter == 3:
            short_task_id += 1
            req_id = f"Short_Ninja_{short_task_id}"
            prompt_short = "Summarize the key points: " * 5
            sampling_params_short = SamplingParams(max_tokens=16)
            
            print(f" [+] 突发短任务: {req_id} 抵达队列!")
            engine.add_request(req_id, prompt_short, sampling_params_short)
            metrics[req_id] = {"arrival": time.perf_counter(), "ttft": None}
        
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
    for req_id, data in metrics.items():
        print(f"{req_id:<20} | {data['ttft']:.2f} ms")

# =====================================================================
# [Action 2 核心代码]：精度与延迟双重验证 (TTFT 高精度探针版)
# =====================================================================
def run_accuracy_verification(engine):
    from vllm.sampling_params import SamplingParams
    import time
    
    print("\n" + "="*60)
    print("🎯 启动模式: 精度与延迟双重验证 (SLA & Accuracy Co-verification)")
    print("="*60)
    
    metrics = {}
    
    # 纯粹的地理常识构建长文本，顺应 Base 模型的续写逻辑
    filler = "London is the capital of the United Kingdom. Berlin is the capital of Germany. Rome is the capital of Italy. Madrid is the capital of Spain. " * 60
    prompt_long = filler + "Tokyo is the capital of Japan. The capital of France is"
    
    sampling_params_long = SamplingParams(max_tokens=32, temperature=0.0)
    
    print("[+] 注入长序列常识补全任务 (Titan)")
    engine.add_request("Long_Titan", prompt_long, sampling_params_long)
    metrics["Long_Titan"] = {"arrival": time.perf_counter(), "ttft": None, "total_time": None, "output": ""}

    step_counter = 0
    
    while engine.has_unfinished_requests():
        step_counter += 1
        
        if step_counter == 2:
            prompt_short_1 = "English: Hello. French: Bonjour. English: Apple. French:"
            sampling_params_short_1 = SamplingParams(max_tokens=32, temperature=0.0)
            print(f"\n [+] 突发短任务 (Tick {step_counter}): 短文本翻译 (Ninja_1) 抵达！")
            engine.add_request("Short_Ninja_1", prompt_short_1, sampling_params_short_1)
            metrics["Short_Ninja_1"] = {"arrival": time.perf_counter(), "ttft": None, "total_time": None, "output": ""}

        if step_counter == 4:
            prompt_short_2 = "10 + 10 = 20. 15 + 25 ="
            sampling_params_short_2 = SamplingParams(max_tokens=32, temperature=0.0)
            print(f"\n [+] 突发短任务 (Tick {step_counter}): 基础数学推理 (Ninja_2) 抵达！")
            engine.add_request("Short_Ninja_2", prompt_short_2, sampling_params_short_2)
            metrics["Short_Ninja_2"] = {"arrival": time.perf_counter(), "ttft": None, "total_time": None, "output": ""}
            
        request_outputs = engine.step()
        current_time = time.perf_counter()
        
        for output in request_outputs:
            req_id = output.request_id
            
            # 【核心修复】：精准捕获首字生成瞬间 (TTFT)
            if metrics[req_id]["ttft"] is None and len(output.outputs[0].token_ids) > 0:
                ttft_ms = (current_time - metrics[req_id]["arrival"]) * 1000 
                metrics[req_id]["ttft"] = ttft_ms
                print(f"  >>> [{req_id}] ⚡ 首字抵达 (TTFT): {ttft_ms:.2f} ms")

            # 捕获全部完成时间
            if output.finished:
                total_ms = (current_time - metrics[req_id]["arrival"]) * 1000 
                metrics[req_id]["total_time"] = total_ms
                
                generated_text = output.outputs[0].text.replace('\n', ' ').strip()
                metrics[req_id]["output"] = generated_text
                
                print(f"  >>> [{req_id}] ✅ 序列生成完毕 | 总耗时: {total_ms:.2f} ms")

    print("\n=== Wave-Slice 精度与延迟双重验证终极报告 ===")
    print(f"{'Task ID':<15} | {'TTFT (ms)':<12} | {'Total Time':<12} | {'Generated Text'}")
    print("-" * 80)
    for req_id, data in metrics.items():
        ttft_str = f"{data['ttft']:.2f}" if data['ttft'] is not None else "TIMEOUT"
        tot_str = f"{data['total_time']:.2f}" if data['total_time'] is not None else "TIMEOUT"
        print(f"{req_id:<15} | {ttft_str:<12} | {tot_str:<12} | '{data['output']}'")

if __name__ == "__main__":
    hf_model_path = "mistralai/Mistral-7B-v0.1"
    engine = setup_engine(hf_model_path, enable_lora=False)
    
    # ---------------------------------------------------------
    # 模式开关：按需取消注释即可运行对应测试
    # ---------------------------------------------------------
    
    # 历史资产：运行微观延迟与调度顺序验证
    # run_latency_benchmark(engine)
    
    # 当前任务 (Action 2)：运行精度无损验证
    run_accuracy_verification(engine)
