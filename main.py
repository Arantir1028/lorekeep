# main.py
import time
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from wave_slice.engine import WaveSliceLLMEngine

def main():
    print("Starting Wave-Slice Pure-Prefill Concurrency Engine...")
    
    engine_args = EngineArgs(
        model="mistralai/Mistral-7B-v0.1", 
        enable_lora=False,
        max_lora_rank=64,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        disable_sliding_window=True
    )
    
    engine = WaveSliceLLMEngine.from_engine_args(engine_args)

    # 构造极端的长任务
    prompt_long = "The quick brown fox " * 600 # 约 3000 Tokens
    sampling_params_long = SamplingParams(max_tokens=1) # 我们只关心 Prefill，生成 1 个字就结束
    engine.add_request("Long_Titan", prompt_long, sampling_params_long)

    print("\n--- Starting Execution Loop (Pure Prefill Interleaving) ---")
    step_counter = 0
    short_task_id = 0
    
    # 只要系统还有任务就一直跑
    while engine.has_unfinished_requests():
        step_counter += 1
        print(f"\n[Tick {step_counter}] Executing Engine Step...")
        
        # 模拟高频短任务的随机到达 (每隔 3 帧注入一个短任务)
        if step_counter % 3 == 1 and step_counter < 15:
            short_task_id += 1
            req_id = f"Short_Ninja_{short_task_id}"
            prompt_short = "Summarize: " * 5 # 极短请求
            sampling_params_short = SamplingParams(max_tokens=1) # 只要拿到 TTFT 就撤
            print(f" [+] Incoming Fast Task: {req_id} arrived at Tick {step_counter}!")
            engine.add_request(req_id, prompt_short, sampling_params_short)
        
        request_outputs = engine.step()
        
        # 解析输出
        for output in request_outputs:
            if output.finished:
                print(f"  >>> [{output.request_id}] TTFT Achieved & Finished!")
            else:
                print(f"  >>> [{output.request_id}] Computing Prefill...")

if __name__ == "__main__":
    main()