
### 主要特点：

1. **双模型支持**：
   - 同时加载 Mistral-7B 和 Falcon-7B 两个模型
   - 使用 `--model1` 和 `--model2` 参数指定模型

2. **三种运行模式**：
   - **串行模式**：先运行模型1的批次，再运行模型2的批次
   - **并行模式**：使用CUDA流同时运行两个模型的批次
   - **批处理模式**：将每个模型的所有请求合并为一个大批次运行

3. **性能统计**：
   - 总时间和平均任务时间
   - 吞吐量（requests/s）
   - GPU利用率、显存利用率和显存使用量
   - 加速比计算

4. **其他功能**：
   - 进度条显示
   - 日志记录
   - 模型预热
   - 支持自定义批次大小和重复次数

### 使用方式：

```bash
python example/hf_two_models_dual_stream.py \
  --model1 mistralai/Mistral-7B-Instruct-v0.2 \
  --model2 tiiuae/falcon-7b-instruct \
  --batch_size 4 \
  --repeats 20 \
  --with_gpu_stats
```

这个脚本会输出详细的性能比较结果，包括三种模式的总时间、平均任务时间、吞吐量和GPU资源使用情况。

```python:example/hf_two_models_dual_stream.py
<code_block_to_apply_changes_from>
import argparse
import time
import os
import threading
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import re

# 设置环境变量
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(1)

class FileLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.log_file = open(file_path, "a")
        
    def log(self, message):
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark two models (Mistral-7B and Falcon-7B) in serial, parallel and batch modes")
    parser.add_argument("--model1", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="First model name")
    parser.add_argument("--model2", type=str, default="tiiuae/falcon-7b-instruct", help="Second model name")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for each model")
    parser.add_argument("--repeats", type=int, default=10, help="Number of times to repeat the benchmark")
    parser.add_argument("--with_gpu_stats", action="store_true", help="Collect GPU statistics during benchmark")
    parser.add_argument("--log_file", type=str, default="hf_two_models_benchmark.log", help="Log file path")
    return parser.parse_args()

def get_torch_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_torch_dtype(),
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_ids(model, tokenizer, input_ids, attention_mask, max_new_tokens, device):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    return outputs

def collect_gpu_stats(device_index=0):
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        
        if not result:
            return 0.0, 0.0, 0.0
        
        # 解析多GPU情况
        gpu_lines = result.split('\n')
        if device_index < len(gpu_lines):
            gpu_line = gpu_lines[device_index]
            gpu_util, mem_used, mem_total = re.findall(r"(\d+\.?\d*)", gpu_line)
            gpu_util = float(gpu_util)
            mem_used = float(mem_used)
            mem_total = float(mem_total)
            mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0.0
            return gpu_util, mem_util, mem_used
        else:
            return 0.0, 0.0, 0.0
    except Exception as e:
        print(f"GPU stats collection failed: {e}")
        return 0.0, 0.0, 0.0

def benchmark_serial(model1, tokenizer1, model2, tokenizer2, input_ids_list1, attention_mask_list1, input_ids_list2, attention_mask_list2, args, device):
    times = []
    gpu_utils = []
    mem_utils = []
    mem_useds = []
    
    for i in tqdm(range(args.repeats), desc="Serial"):
        start_time = time.time()
        
        # 模型1的批次
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            generate_ids(model1, tokenizer1, input_ids_list1[idx], attention_mask_list1[idx], args.max_new_tokens, device)
        
        # 模型2的批次
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            generate_ids(model2, tokenizer2, input_ids_list2[idx], attention_mask_list2[idx], args.max_new_tokens, device)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        if args.with_gpu_stats:
            gpu_util, mem_util, mem_used = collect_gpu_stats()
            gpu_utils.append(gpu_util)
            mem_utils.append(mem_util)
            mem_useds.append(mem_used)
    
    return times, gpu_utils, mem_utils, mem_useds

def _thread_target_stream(model, tokenizer, input_ids, attention_mask, max_new_tokens, device, stream):
    with torch.cuda.stream(stream):
        generate_ids(model, tokenizer, input_ids, attention_mask, max_new_tokens, device)

def benchmark_parallel_streams(model1, tokenizer1, model2, tokenizer2, input_ids_list1, attention_mask_list1, input_ids_list2, attention_mask_list2, args, device):
    times = []
    gpu_utils = []
    mem_utils = []
    mem_useds = []
    
    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)
    
    for i in tqdm(range(args.repeats), desc="Parallel Streams"):
        start_time = time.time()
        
        threads = []
        # 模型1的批次
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            t = threading.Thread(
                target=_thread_target_stream,
                args=(model1, tokenizer1, input_ids_list1[idx], attention_mask_list1[idx], args.max_new_tokens, device, stream1)
            )
            threads.append(t)
            t.start()
        
        # 模型2的批次
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            t = threading.Thread(
                target=_thread_target_stream,
                args=(model2, tokenizer2, input_ids_list2[idx], attention_mask_list2[idx], args.max_new_tokens, device, stream2)
            )
            threads.append(t)
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        if args.with_gpu_stats:
            gpu_util, mem_util, mem_used = collect_gpu_stats()
            gpu_utils.append(gpu_util)
            mem_utils.append(mem_util)
            mem_useds.append(mem_used)
    
    return times, gpu_utils, mem_utils, mem_useds

def benchmark_big_batch(model1, tokenizer1, model2, tokenizer2, input_ids_big1, attention_mask_big1, input_ids_big2, attention_mask_big2, args, device):
    times = []
    gpu_utils = []
    mem_utils = []
    mem_useds = []
    
    for i in tqdm(range(args.repeats), desc="Big Batch"):
        start_time = time.time()
        
        # 模型1的大批次
        generate_ids(model1, tokenizer1, input_ids_big1, attention_mask_big1, args.max_new_tokens, device)
        
        # 模型2的大批次
        generate_ids(model2, tokenizer2, input_ids_big2, attention_mask_big2, args.max_new_tokens, device)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        if args.with_gpu_stats:
            gpu_util, mem_util, mem_used = collect_gpu_stats()
            gpu_utils.append(gpu_util)
            mem_utils.append(mem_util)
            mem_useds.append(mem_used)
    
    return times, gpu_utils, mem_utils, mem_useds

def main():
    args = parse_args()
    logger = FileLogger(args.log_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.log(f"Using device: {device}")
    logger.log(f"Model 1: {args.model1}")
    logger.log(f"Model 2: {args.model2}")
    logger.log(f"Batch size: {args.batch_size}, Repeats: {args.repeats}")
    
    # 加载两个模型
    logger.log("Loading model 1...")
    model1, tokenizer1 = load_model_and_tokenizer(args.model1, device)
    logger.log("Loading model 2...")
    model2, tokenizer2 = load_model_and_tokenizer(args.model2, device)
    
    # 准备提示
    prompts_batch1 = ["What is the capital of France?"] * (args.repeats * args.batch_size)
    prompts_batch2 = ["Explain quantum computing in simple terms"] * (args.repeats * args.batch_size)
    
    # 为每个模型单独tokenize
    logger.log("Tokenizing prompts for model 1...")
    tokenized1 = tokenizer1(prompts_batch1, return_tensors="pt", padding=True)
    input_ids_list1 = tokenized1["input_ids"].split(1)
    attention_mask_list1 = tokenized1["attention_mask"].split(1)
    
    logger.log("Tokenizing prompts for model 2...")
    tokenized2 = tokenizer2(prompts_batch2, return_tensors="pt", padding=True)
    input_ids_list2 = tokenized2["input_ids"].split(1)
    attention_mask_list2 = tokenized2["attention_mask"].split(1)
    
    # 准备大批次数据
    input_ids_big1 = tokenized1["input_ids"]
    attention_mask_big1 = tokenized1["attention_mask"]
    input_ids_big2 = tokenized2["input_ids"]
    attention_mask_big2 = tokenized2["attention_mask"]
    
    # 预热
    logger.log("Warming up models...")
    warmup_prompt = "Warmup"
    warmup_input1 = tokenizer1(warmup_prompt, return_tensors="pt").to(device)
    warmup_input2 = tokenizer2(warmup_prompt, return_tensors="pt").to(device)
    generate_ids(model1, tokenizer1, warmup_input1.input_ids, warmup_input1.attention_mask, 10, device)
    generate_ids(model2, tokenizer2, warmup_input2.input_ids, warmup_input2.attention_mask, 10, device)
    torch.cuda.synchronize()
    
    # 运行基准测试
    logger.log("Starting serial benchmark...")
    serial_times, serial_gpu_utils, serial_mem_utils, serial_mem_useds = benchmark_serial(
        model1, tokenizer1, model2, tokenizer2, 
        input_ids_list1, attention_mask_list1, 
        input_ids_list2, attention_mask_list2, 
        args, device
    )
    
    logger.log("Starting parallel streams benchmark...")
    parallel_times, parallel_gpu_utils, parallel_mem_utils, parallel_mem_useds = benchmark_parallel_streams(
        model1, tokenizer1, model2, tokenizer2, 
        input_ids_list1, attention_mask_list1, 
        input_ids_list2, attention_mask_list2, 
        args, device
    )
    
    logger.log("Starting big batch benchmark...")
    bigbatch_times, bigbatch_gpu_utils, bigbatch_mem_utils, bigbatch_mem_useds = benchmark_big_batch(
        model1, tokenizer1, model2, tokenizer2, 
        input_ids_big1, attention_mask_big1, 
        input_ids_big2, attention_mask_big2, 
        args, device
    )
    
    # 计算统计数据
    def calc_stats(times):
        total = sum(times)
        avg = total / len(times)
        return total, avg
    
    serial_total, serial_avg = calc_stats(serial_times)
    parallel_total, parallel_avg = calc_stats(parallel_times)
    bigbatch_total, bigbatch_avg = calc_stats(bigbatch_times)
    
    # 计算吞吐量 (requests/s)
    total_requests = args.repeats * args.batch_size * 2  # 两个模型
    serial_throughput = total_requests / serial_total
    parallel_throughput = total_requests / parallel_total
    bigbatch_throughput = total_requests / bigbatch_total
    
    # 计算平均GPU统计
    def avg_gpu_stats(stats):
        return sum(stats) / len(stats) if stats else 0.0
    
    serial_gpu_util = avg_gpu_stats(serial_gpu_utils)
    serial_mem_util = avg_gpu_stats(serial_mem_utils)
    serial_mem_used = avg_gpu_stats(serial_mem_useds)
    
    parallel_gpu_util = avg_gpu_stats(parallel_gpu_utils)
    parallel_mem_util = avg_gpu_stats(parallel_mem_utils)
    parallel_mem_used = avg_gpu_stats(parallel_mem_useds)
    
    bigbatch_gpu_util = avg_gpu_stats(bigbatch_gpu_utils)
    bigbatch_mem_util = avg_gpu_stats(bigbatch_mem_utils)
    bigbatch_mem_used = avg_gpu_stats(bigbatch_mem_useds)
    
    # 结果比较
    comparison = (
        f"Serial: total={serial_total:.4f}s, avg_task={serial_avg:.4f}s, throughput={serial_throughput:.2f} req/s, "
        f"gpu_util={serial_gpu_util:.1f}%, mem_util={serial_mem_util:.1f}%, mem_used={serial_mem_used:.1f} MiB | "
        f"Streams: total={parallel_total:.4f}s, avg_task={parallel_avg:.4f}s, throughput={parallel_throughput:.2f} req/s, "
        f"gpu_util={parallel_gpu_util:.1f}%, mem_util={parallel_mem_util:.1f}%, mem_used={parallel_mem_used:.1f} MiB | "
        f"BigBatch: total={bigbatch_total:.4f}s, avg_task={bigbatch_avg:.4f}s, throughput={bigbatch_throughput:.2f} req/s, "
        f"gpu_util={bigbatch_gpu_util:.1f}%, mem_util={bigbatch_mem_util:.1f}%, mem_used={bigbatch_mem_used:.1f} MiB"
    )
    
    logger.log("\nPerformance Comparison:")
    logger.log(comparison)
    
    # 计算加速比
    serial_speedup = serial_total / parallel_total
    bigbatch_speedup = serial_total / bigbatch_total
    
    logger.log(f"\nSpeedup vs Serial:")
    logger.log(f"  Parallel Streams: {serial_speedup:.2f}x")
    logger.log(f"  Big Batch: {bigbatch_speedup:.2f}x")
    
    logger.close()

if __name__ == "__main__":
    main()
