'''
# 请求同时到达，两个模型，并行执行
python example/hf_two_models_dual_stream.py   --cuda_visible_devices 1     --max_new_tokens 32 --repeats 100 \
    --batch_size 0 --with_gpu_stats     --log_file example/hf_two_models_dual_stream.log --generate_plot
'''
import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 解析命令行参数获取 GPU 设置
cuda_visible_devices = '1'  # 默认值
for i, arg in enumerate(sys.argv):
    if arg == '--cuda_visible_devices' and i + 1 < len(sys.argv):
        cuda_visible_devices = sys.argv[i + 1]
        break
    elif arg.startswith('--cuda_visible_devices='):
        cuda_visible_devices = arg.split('=')[1]
        break

# 在导入任何 PyTorch 模块前设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# 现在导入其他模块
import argparse
import time
import threading
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import re
import numpy as np # Added for np.mean
import multiprocessing as mp
import matplotlib.pyplot as plt
import csv  # 添加csv模块导入


# 在文件顶部定义全局logger
logger = None

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
    parser = argparse.ArgumentParser(description="双模型并行推理基准测试")
    parser.add_argument("--model1", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="First model name")
    parser.add_argument("--model2", type=str, default="tiiuae/falcon-7b-instruct", help="Second model name")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=0, 
                        help="单个batch size测试(0表示测试所有batch size)")
    parser.add_argument("--repeats", type=int, default=100, help="Number of times to repeat the benchmark")
    parser.add_argument("--with_gpu_stats", action="store_true", help="Collect GPU statistics during benchmark")
    parser.add_argument("--log_file", type=str, default="hf_two_models_benchmark.log", help="Log file path")
    parser.add_argument("--cuda_visible_devices", type=str, default="1", help="Physical GPU index to use")
    parser.add_argument("--generate_plot", action="store_true", 
                        help="生成性能对比图表")
    return parser.parse_args()

def get_torch_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def load_model_and_tokenizer(model_name, args, device):
    """加载模型和tokenizer，并移动到指定设备"""
    # 获取torch数据类型
    torch_dtype = get_torch_dtype()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": device},
        # trust_remote_code=True
    )
    model.eval()
    
    return tokenizer, model

# 移除batch_merge_generate函数
# 这个函数不适用于不同模型并行的情况

# 保留优化后的manual_generate函数
def manual_generate(model, input_ids, attention_mask, max_new_tokens, device, stream=None, chunk_size=4):
    """手动实现chunk解码循环，支持CUDA流"""
    # 确保输入在正确的设备上
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    
    # 设置生成上下文
    if stream is not None:
        original_stream = torch.cuda.current_stream(device)
        torch.cuda.set_stream(stream)
    
    # 初始化缓存
    past_key_values = None
    generated_tokens = input_ids.clone()
    chunk_times = []  # 存储每个chunk的耗时
    
    with torch.no_grad():
        # 计算需要多少次chunk解码
        num_chunks = (max_new_tokens + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            chunk_start = time.time()
            # 计算当前chunk要生成的token数
            tokens_to_generate = min(chunk_size, max_new_tokens - chunk_idx * chunk_size)
            
            # 准备输入 - 只使用最后一个token
            if past_key_values is not None:
                input_ids = generated_tokens[:, -1].unsqueeze(-1)
            
            # 创建模型输入
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": True
            }
            
            # 在指定流上执行前向传播
            outputs = model(**model_inputs)
            
            # 获取下一个token的logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # 一次生成多个token
            for i in range(tokens_to_generate):
                # 获取下一个token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 更新生成结果
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                
                # 更新注意力掩码
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)
                
                # 如果是chunk内的后续token，准备下一次输入
                if i < tokens_to_generate - 1:
                    # 准备输入 - 只使用最后一个token
                    input_ids = next_token
                    
                    # 创建模型输入
                    model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "past_key_values": outputs.past_key_values,
                        "use_cache": True
                    }
                    
                    # 在指定流上执行前向传播
                    outputs = model(**model_inputs)
                    next_token_logits = outputs.logits[:, -1, :]
            
            # 更新缓存
            past_key_values = outputs.past_key_values
            chunk_elapsed = time.time() - chunk_start
            chunk_times.append(chunk_elapsed)
    
    # 恢复原始流
    if stream is not None:
        torch.cuda.set_stream(original_stream)
    
    return generated_tokens, chunk_times

# 保留优化后的模型线程函数
def model_thread(model, input_ids, attention_mask, max_new_tokens, device, timing_list):
    """模型线程函数，记录每个请求平均时间"""
    # 创建独立的CUDA流
    stream = torch.cuda.Stream(device=device)
    
    # 记录实际开始时间
    start_time = time.time()
    
    with torch.cuda.stream(stream):
        # 执行模型推理
        outputs = manual_generate(model, input_ids, attention_mask, max_new_tokens, device)
    
    # 确保流完成
    stream.synchronize()
    
    # 记录实际执行时间
    elapsed = time.time() - start_time
    
    # 计算batch内平均请求时间
    batch_size = input_ids.shape[0] if input_ids is not None else 0
    if batch_size > 0:
        per_request_time = elapsed / batch_size
        timing_list.append(per_request_time)  # 保存平均请求时间
    else:
        timing_list.append(0.0)  # 异常情况返回0

def get_physical_gpu_index(cuda_visible_devices):
    """根据CUDA_VISIBLE_DEVICES设置获取实际物理GPU索引"""
    # 用户特定映射规则：
    # --cuda_visible_devices 0 → 实际使用物理GPU1 → 监控GPU1
    # --cuda_visible_devices 1 → 实际使用物理GPU0 → 监控GPU0
    return 1 - int(cuda_visible_devices)

def gpu_monitor(stop_event, interval, stats, physical_device_index):
    """GPU监控线程函数，使用nvidia-smi收集实时数据"""
    while not stop_event.is_set():
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi", 
                    "-i", str(physical_device_index),
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                    "--format=csv,noheader,nounits"
                ],
                encoding="utf-8"
            )
            
            util_gpu, util_mem, mem_used = map(float, result.strip().split(","))
            
            stats["gpu"].append(util_gpu)
            stats["mem"].append(util_mem)
            stats["mem_used"].append(mem_used)
            
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        
        time.sleep(interval)

def benchmark_serial(model1, model2, 
                    input_ids_batches1, attention_mask_batches1,
                    input_ids_batches2, attention_mask_batches2,
                    args, device, stats=None):
    """串行基准测试函数，记录每个请求的时间，支持GPU监控"""
    times = []
    model1_times = []  # 存储模型1的实际执行时间
    model2_times = []  # 存储模型2的实际执行时间
    request_times = []  # 存储每个请求的完成时间
    
    num_batches = len(input_ids_batches1)
    if num_batches == 0:
        logger.log("Warning: No batches available for serial benchmark (batch_size may be too large)")
        return times, request_times, model1_times, model2_times
    
    # 如果启用GPU统计，启动监控线程
    stop_event = threading.Event() if stats else None
    monitor = None
    if stats:
        physical_device_index = get_physical_gpu_index(args.cuda_visible_devices)
        monitor = threading.Thread(
            target=gpu_monitor, 
            args=(stop_event, 0.05, stats, physical_device_index)
        )
        monitor.start()
    
    for i in tqdm(range(args.repeats), desc="Serial"):
        batch_idx = i % num_batches  # 循环使用可用批次
        
        batch_start = time.time()
        
        # 执行模型1并记录时间
        model1_start = time.time()
        outputs1 = manual_generate(model1, input_ids_batches1[batch_idx], attention_mask_batches1[batch_idx], 
                                  args.max_new_tokens, device)
        model1_elapsed = time.time() - model1_start
        model1_times.append(model1_elapsed)
        
        # 执行模型2并记录时间
        model2_start = time.time()
        outputs2 = manual_generate(model2, input_ids_batches2[batch_idx], attention_mask_batches2[batch_idx], 
                                  args.max_new_tokens, device)
        model2_elapsed = time.time() - model2_start
        model2_times.append(model2_elapsed)
        
        batch_elapsed = time.time() - batch_start
        times.append(batch_elapsed)
        
        # 计算每个请求的完成时间（添加除零保护）
        if args.batch_size > 0:
            batch_avg_request_time = batch_elapsed / (args.batch_size * 2)
            request_times.extend([batch_avg_request_time] * (args.batch_size * 2))
    
    # 停止GPU监控
    if monitor:
        stop_event.set()
        monitor.join()
    
    return times, request_times, model1_times, model2_times

def benchmark_parallel_threads(model1, model2, 
                              input_ids_batches1, attention_mask_batches1,
                              input_ids_batches2, attention_mask_batches2,
                              args, device, stats=None):
    """并行基准测试函数，记录每个请求的时间，支持GPU监控"""
    times = []
    model1_times = []  # 存储模型1的实际执行时间
    model2_times = []  # 存储模型2的实际执行时间
    request_times = []  # 存储每个请求的完成时间
    
    num_batches = len(input_ids_batches1)
    if num_batches == 0:
        logger.log("Warning: No batches available for parallel benchmark (batch_size may be too large)")
        return times, request_times, model1_times, model2_times
    
    # 如果启用GPU统计，启动监控线程
    stop_event = threading.Event() if stats else None
    monitor = None
    if stats:
        physical_device_index = get_physical_gpu_index(args.cuda_visible_devices)
        monitor = threading.Thread(
            target=gpu_monitor, 
            args=(stop_event, 0.05, stats, physical_device_index)
        )
        monitor.start()
    
    for i in tqdm(range(args.repeats), desc="Parallel Threads"):
        batch_idx = i % num_batches  # 循环使用可用批次
        
        batch_start = time.time()
        
        # 准备时间记录列表
        thread1_times = []  # 用于存储模型1平均请求时间
        thread2_times = []  # 用于存储模型2平均请求时间
        
        # 创建两个线程分别处理两个模型
        thread1 = threading.Thread(
            target=model_thread, 
            args=(model1, input_ids_batches1[batch_idx], attention_mask_batches1[batch_idx], 
                 args.max_new_tokens, device, thread1_times)
        )
        thread2 = threading.Thread(
            target=model_thread, 
            args=(model2, input_ids_batches2[batch_idx], attention_mask_batches2[batch_idx], 
                 args.max_new_tokens, device, thread2_times)
        )
        
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        
        batch_elapsed = time.time() - batch_start
        times.append(batch_elapsed)
        
        # 存储模型实际执行时间
        model1_actual = thread1_times[0] if thread1_times else 0.0
        model2_actual = thread2_times[0] if thread2_times else 0.0
        model1_times.append(model1_actual)
        model2_times.append(model2_actual)
        
        # 计算batch内所有请求的平均时间（使用总时间分摊）
        if args.batch_size > 0:
            batch_avg_request_time = batch_elapsed / (args.batch_size * 2)
            request_times.extend([batch_avg_request_time] * (args.batch_size * 2))
    
    # 停止GPU监控
    if monitor:
        stop_event.set()
        monitor.join()
    
    return times, request_times, model1_times, model2_times

# 7. 上下文管理器用于处理无流情况
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def calculate_percentage_change(serial_value, parallel_value):
    """计算并行相对于串行的性能变化百分比"""
    if serial_value == 0:
        return 0.0
    return ((parallel_value - serial_value) / serial_value) * 100.0

def generate_performance_plot(serial_time, parallel_time, 
                             serial_throughput, parallel_throughput,
                             serial_stats, parallel_stats, 
                             output_path):
    """Generate performance comparison plot"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Serial vs Parallel Performance Comparison', fontsize=16)
    
    # 1. Time comparison
    times = [serial_time, parallel_time]
    labels = ['Serial', 'Parallel']
    colors = ['skyblue', 'orange']  # 更新颜色方案
    
    axs[0, 0].bar(labels, times, color=colors)
    axs[0, 0].set_title('Average Time Comparison')
    axs[0, 0].set_ylabel('Time (seconds)')
    
    # Add value labels
    for i, v in enumerate(times):
        axs[0, 0].text(i, v + 0.1 * max(times), f"{v:.4f}", 
                      ha='center', fontweight='bold')
    
    # 2. Throughput comparison
    throughputs = [serial_throughput, parallel_throughput]
    
    axs[0, 1].bar(labels, throughputs, color=colors)  # 使用相同颜色方案
    axs[0, 1].set_title('Throughput Comparison')
    axs[0, 1].set_ylabel('Samples per second')
    
    # Add value labels
    for i, v in enumerate(throughputs):
        axs[0, 1].text(i, v + 0.1 * max(throughputs), f"{v:.2f}", 
                      ha='center', fontweight='bold')
    
    # 3. GPU utilization comparison (if available)
    if serial_stats and parallel_stats:
        gpu_utils = [
            np.mean(serial_stats['gpu']), 
            np.mean(parallel_stats['gpu'])
        ]
        
        axs[1, 0].bar(labels, gpu_utils, color=colors)  # 使用相同颜色方案
        axs[1, 0].set_title('GPU Utilization Comparison')
        axs[1, 0].set_ylabel('Utilization (%)')
        axs[1, 0].set_ylim(0, 100)  # Fixed 0-100% range
        
        # Add value labels
        for i, v in enumerate(gpu_utils):
            axs[1, 0].text(i, v + 2, f"{v:.2f}%", 
                          ha='center', fontweight='bold')
    
    # 4. Memory utilization comparison (if available)
    if serial_stats and parallel_stats:
        mem_utils = [
            np.mean(serial_stats['mem']), 
            np.mean(parallel_stats['mem'])
        ]
        
        axs[1, 1].bar(labels, mem_utils, color=colors)  # 使用相同颜色方案
        axs[1, 1].set_title('Memory Utilization Comparison')
        axs[1, 1].set_ylabel('Utilization (%)')
        axs[1, 1].set_ylim(0, 100)  # Fixed 0-100% range
        
        # Add value labels
        for i, v in enumerate(mem_utils):
            axs[1, 1].text(i, v + 2, f"{v:.2f}%", 
                          ha='center', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"Performance comparison plot saved to: {output_path}")

def prepare_inputs(tokenizer, prompts, args, device):
    """准备输入数据"""
    input_ids_batches = []
    attention_mask_batches = []
    
    # 分批处理提示词
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        inputs = tokenizer(
            batch_prompts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        input_ids_batches.append(inputs.input_ids.to(device))
        attention_mask_batches.append(inputs.attention_mask.to(device))
    
    return input_ids_batches, attention_mask_batches

def generate_batch_size_plot(results, output_path):
    """Generate batch size performance plot with all metrics"""
    fig, axs = plt.subplots(4, 2, figsize=(15, 24))  # 4x2布局
    fig.suptitle('Performance by Batch Size', fontsize=16)
    
    batch_sizes = results['batch_sizes']
    
    # 统一设置x轴刻度
    def set_batch_size_xticks(ax):
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels(batch_sizes)
        ax.set_xscale('log')
    
    # 1. Total Completion Time
    axs[0, 0].plot(batch_sizes, results['serial_total_times'], 'o-', color='skyblue', label='Serial')
    axs[0, 0].plot(batch_sizes, results['parallel_total_times'], 'o-', color='orange', label='Parallel')
    axs[0, 0].set_title('Total Completion Time')
    axs[0, 0].set_xlabel('Batch Size')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].grid(True, which="both", ls="--")
    axs[0, 0].legend()
    set_batch_size_xticks(axs[0, 0])
    
    # 2. Average Request Time
    axs[0, 1].plot(batch_sizes, results['serial_avg_request_times'], 'o-', color='skyblue', label='Serial')
    axs[0, 1].plot(batch_sizes, results['parallel_avg_request_times'], 'o-', color='orange', label='Parallel')
    axs[0, 1].set_title('Average Request Time')
    axs[0, 1].set_xlabel('Batch Size')
    axs[0, 1].set_ylabel('Time (seconds)')
    axs[0, 1].grid(True, which="both", ls="--")
    axs[0, 1].legend()
    set_batch_size_xticks(axs[0, 1])
    
    # 3. Throughput
    axs[1, 0].plot(batch_sizes, results['serial_throughputs'], 'o-', color='skyblue', label='Serial')
    axs[1, 0].plot(batch_sizes, results['parallel_throughputs'], 'o-', color='orange', label='Parallel')
    axs[1, 0].set_title('Throughput')
    axs[1, 0].set_xlabel('Batch Size')
    axs[1, 0].set_ylabel('Samples per second')
    axs[1, 0].grid(True, which="both", ls="--")
    axs[1, 0].legend()
    set_batch_size_xticks(axs[1, 0])
    
    # 4. GPU Utilization (if available)
    if results['serial_gpu_utils']:
        axs[1, 1].plot(batch_sizes, results['serial_gpu_utils'], 'o-', color='skyblue', label='Serial')
        axs[1, 1].plot(batch_sizes, results['parallel_gpu_utils'], 'o-', color='orange', label='Parallel')
        axs[1, 1].set_title('GPU Utilization')
        axs[1, 1].set_xlabel('Batch Size')
        axs[1, 1].set_ylabel('Utilization (%)')
        axs[1, 1].set_ylim(0, 100)
        axs[1, 1].grid(True, which="both", ls="--")
        axs[1, 1].legend()
        set_batch_size_xticks(axs[1, 1])
    else:
        axs[1, 1].text(0.5, 0.5, 'No GPU Data Available', ha='center', va='center')
        axs[1, 1].axis('off')
    
    # 5. Memory Utilization (if available)
    if results['serial_mem_utils']:
        axs[2, 0].plot(batch_sizes, results['serial_mem_utils'], 'o-', color='skyblue', label='Serial')
        axs[2, 0].plot(batch_sizes, results['parallel_mem_utils'], 'o-', color='orange', label='Parallel')
        axs[2, 0].set_title('Memory Utilization')
        axs[2, 0].set_xlabel('Batch Size')
        axs[2, 0].set_ylabel('Utilization (%)')
        axs[2, 0].set_ylim(0, 100)
        axs[2, 0].grid(True, which="both", ls="--")
        axs[2, 0].legend()
        set_batch_size_xticks(axs[2, 0])
    else:
        axs[2, 0].text(0.5, 0.5, 'No Memory Data Available', ha='center', va='center')
        axs[2, 0].axis('off')
    
    # 6. Memory Used (if available)
    if results['serial_mem_used']:
        axs[2, 1].plot(batch_sizes, results['serial_mem_used'], 'o-', color='skyblue', label='Serial')
        axs[2, 1].plot(batch_sizes, results['parallel_mem_used'], 'o-', color='orange', label='Parallel')
        axs[2, 1].set_title('Memory Used')
        axs[2, 1].set_xlabel('Batch Size')
        axs[2, 1].set_ylabel('Memory (MB)')
        axs[2, 1].grid(True, which="both", ls="--")
        axs[2, 1].legend()
        set_batch_size_xticks(axs[2, 1])
    else:
        axs[2, 1].text(0.5, 0.5, 'No Memory Used Data Available', ha='center', va='center')
        axs[2, 1].axis('off')
    
    # 7. Model Actual Execution Time (combined)
    axs[3, 0].plot(batch_sizes, results['serial_model1_times'], 'o-', color='skyblue', label='Serial Model1')
    axs[3, 0].plot(batch_sizes, results['parallel_model1_times'], 'o-', color='orange', label='Parallel Model1')
    axs[3, 0].plot(batch_sizes, results['serial_model2_times'], 'o--', color='skyblue', label='Serial Model2')
    axs[3, 0].plot(batch_sizes, results['parallel_model2_times'], 'o--', color='orange', label='Parallel Model2')
    axs[3, 0].set_title('Model Actual Execution Time')
    axs[3, 0].set_xlabel('Batch Size')
    axs[3, 0].set_ylabel('Time (seconds)')
    axs[3, 0].grid(True, which="both", ls="--")
    axs[3, 0].legend()
    set_batch_size_xticks(axs[3, 0])
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"Batch size performance plot saved to: {output_path}")

def print_summary_table(results, logger, log_file):
    """生成并打印所有 batch_size 的汇总表格，并输出CSV"""
    batch_sizes = results['batch_sizes']
    
    # 表格头
    headers = [
        "Batch Size",
        "Serial Total Time (s)",
        "Parallel Total Time (s)",
        "Serial Avg Req Time (s)",
        "Parallel Avg Req Time (s)",
        "Serial Throughput (samples/s)",
        "Parallel Throughput (samples/s)",
        "Serial Model1 Time (s)",
        "Parallel Model1 Time (s)",
        "Serial Model2 Time (s)",
        "Parallel Model2 Time (s)",
        "Serial GPU Util (%)",
        "Parallel GPU Util (%)",
        "Serial Mem Util (%)",
        "Parallel Mem Util (%)",
        "Serial Mem Used (MB)",
        "Parallel Mem Used (MB)"
    ]
    
    # 表格行
    table_rows = []
    for i, bs in enumerate(batch_sizes):
        row = [
            bs,
            f"{results['serial_total_times'][i]:.4f}",
            f"{results['parallel_total_times'][i]:.4f}",
            f"{results['serial_avg_request_times'][i]:.6f}",
            f"{results['parallel_avg_request_times'][i]:.6f}",
            f"{results['serial_throughputs'][i]:.2f}",
            f"{results['parallel_throughputs'][i]:.2f}",
            f"{results['serial_model1_times'][i]:.4f}",
            f"{results['parallel_model1_times'][i]:.4f}",
            f"{results['serial_model2_times'][i]:.4f}",
            f"{results['parallel_model2_times'][i]:.4f}",
            f"{results['serial_gpu_utils'][i]:.2f}" if results['serial_gpu_utils'] else "N/A",
            f"{results['parallel_gpu_utils'][i]:.2f}" if results['parallel_gpu_utils'] else "N/A",
            f"{results['serial_mem_utils'][i]:.2f}" if results['serial_mem_utils'] else "N/A",
            f"{results['parallel_mem_utils'][i]:.2f}" if results['parallel_mem_utils'] else "N/A",
            f"{results['serial_mem_used'][i]:.2f}" if results['serial_mem_used'] else "N/A",
            f"{results['parallel_mem_used'][i]:.2f}" if results['parallel_mem_used'] else "N/A"
        ]
        table_rows.append(row)
    
    # 生成ASCII表格
    col_widths = [max(len(str(row[j])) for row in table_rows + [headers]) for j in range(len(headers))]
    
    def format_row(row):
        return " | ".join(f"{item:<{col_widths[j]}}" for j, item in enumerate(row))
    
    table_str = ""
    table_str += format_row(headers) + "\n"
    table_str += "-" * (sum(col_widths) + len(headers) * 3 - 1) + "\n"
    for row in table_rows:
        table_str += format_row(row) + "\n"
    
    # 输出到日志和控制台
    logger.log(table_str)
    print(table_str)
    
    # 生成CSV文件
    csv_path = log_file.replace(".log", ".csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(table_rows)
    
    logger.log(f"Summary table saved as CSV to: {csv_path}")
    print(f"Summary table saved as CSV to: {csv_path}")

def main(args):
    global logger
    
    # 设置设备
    device = torch.device(f"cuda:{args.cuda_visible_devices}" if torch.cuda.is_available() else "cpu")
    
    # 定义要测试的batch size列表
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128] if args.batch_size == 0 else [args.batch_size]
    # batch_sizes = [1, 2, 4] if args.batch_size == 0 else [args.batch_size]
    
    # 结果存储结构（添加mem_used）
    results = {
        'batch_sizes': batch_sizes,
        'serial_total_times': [],        # 串行总完成时间
        'parallel_total_times': [],      # 并行总完成时间
        'serial_avg_request_times': [],  # 串行平均请求时间
        'parallel_avg_request_times': [],# 并行平均请求时间
        'serial_throughputs': [],
        'parallel_throughputs': [],
        'serial_gpu_utils': [],
        'parallel_gpu_utils': [],
        'serial_mem_utils': [],
        'parallel_mem_utils': [],
        'serial_model1_times': [],    # 串行模式下模型1实际执行时间
        'serial_model2_times': [],    # 串行模式下模型2实际执行时间
        'parallel_model1_times': [],  # 并行模式下模型1实际执行时间
        'parallel_model2_times': [],   # 并行模式下模型2实际执行时间
        'serial_mem_used': [],   # 新增显存使用量
        'parallel_mem_used': []  # 新增显存使用量
    }
    
    # 加载模型（只加载一次）
    logger.log("Loading models...")
    tokenizer1, model1 = load_model_and_tokenizer(args.model1, args, device)
    tokenizer2, model2 = load_model_and_tokenizer(args.model2, args, device)
    
    # 准备提示词（只准备一次）
    prompts1 = [f"Explain quantum computing in simple terms {i}" for i in range(100)]
    prompts2 = [f"Describe the theory of relativity {i}" for i in range(100)]
    
    # 循环测试所有batch size
    for batch_size in tqdm(batch_sizes, desc="Testing batch sizes"):
        logger.log(f"\n{'='*40}")
        logger.log(f"Testing with batch size: {batch_size}")
        logger.log(f"{'='*40}")
        
        # 更新当前batch size
        current_args = argparse.Namespace(**vars(args))
        current_args.batch_size = batch_size
        
        # 准备输入数据
        input_ids_batches1, attention_mask_batches1 = prepare_inputs(
            tokenizer1, prompts1, current_args, device
        )
        input_ids_batches2, attention_mask_batches2 = prepare_inputs(
            tokenizer2, prompts2, current_args, device
        )
        
        # 初始化GPU统计变量
        serial_stats = {"gpu": [], "mem": [], "mem_used": []} if current_args.with_gpu_stats else None
        parallel_stats = {"gpu": [], "mem": [], "mem_used": []} if current_args.with_gpu_stats else None
        
        # 运行串行基准测试
        serial_times, serial_request_times, serial_model1_times, serial_model2_times = benchmark_serial(
            model1, model2,
            input_ids_batches1, attention_mask_batches1,
            input_ids_batches2, attention_mask_batches2,
            current_args, device, serial_stats
        )
        
        # 运行并行基准测试
        parallel_times, parallel_request_times, parallel_model1_times, parallel_model2_times = benchmark_parallel_threads(
            model1, model2,
            input_ids_batches1, attention_mask_batches1,
            input_ids_batches2, attention_mask_batches2,
            current_args, device, parallel_stats
        )
        
        # 计算指标
        total_requests = current_args.repeats * current_args.batch_size * 2
        
        # 总完成时间
        total_serial_time = sum(serial_times)
        total_parallel_time = sum(parallel_times)
        
        # 平均请求时间（基于实际测量）
        avg_request_time_serial = np.mean(serial_request_times) if serial_request_times else 0
        avg_request_time_parallel = np.mean(parallel_request_times) if parallel_request_times else 0
        
        # 吞吐量
        serial_throughput = total_requests / total_serial_time if total_serial_time > 0 else 0
        parallel_throughput = total_requests / total_parallel_time if total_parallel_time > 0 else 0
        
        # 存储结果
        results['serial_total_times'].append(total_serial_time)
        results['parallel_total_times'].append(total_parallel_time)
        results['serial_avg_request_times'].append(avg_request_time_serial)
        results['parallel_avg_request_times'].append(avg_request_time_parallel)
        results['serial_model1_times'].append(np.mean(serial_model1_times))
        results['serial_model2_times'].append(np.mean(serial_model2_times))
        results['parallel_model1_times'].append(np.mean(parallel_model1_times))
        results['parallel_model2_times'].append(np.mean(parallel_model2_times))
        results['serial_throughputs'].append(serial_throughput)
        results['parallel_throughputs'].append(parallel_throughput)
        
        if current_args.with_gpu_stats:
            results['serial_gpu_utils'].append(np.mean(serial_stats['gpu']))
            results['parallel_gpu_utils'].append(np.mean(parallel_stats['gpu']))
            results['serial_mem_utils'].append(np.mean(serial_stats['mem']))
            results['parallel_mem_utils'].append(np.mean(parallel_stats['mem']))
            results['serial_mem_used'].append(np.mean(serial_stats['mem_used']))
            results['parallel_mem_used'].append(np.mean(parallel_stats['mem_used']))
        
        # 记录当前batch size结果（保持原有输出）
        logger.log(f"Batch size {batch_size} results:")
        logger.log(f"  Serial total time: {total_serial_time:.4f}s")
        logger.log(f"  Parallel total time: {total_parallel_time:.4f}s")
        logger.log(f"  Serial avg request time (all requests in batch): {avg_request_time_serial:.6f}s")
        logger.log(f"  Parallel avg request time (all requests in batch): {avg_request_time_parallel:.6f}s")
        logger.log(f"  Serial throughput: {serial_throughput:.2f} samples/s")
        logger.log(f"  Parallel throughput: {parallel_throughput:.2f} samples/s")
        logger.log(f"  Serial model1 actual time: {np.mean(serial_model1_times):.4f}s")
        logger.log(f"  Parallel model1 actual time: {np.mean(parallel_model1_times):.4f}s")
        logger.log(f"  Serial model2 actual time: {np.mean(serial_model2_times):.4f}s")
        logger.log(f"  Parallel model2 actual time: {np.mean(parallel_model2_times):.4f}s")
        
        if current_args.with_gpu_stats:
            logger.log(f"  Serial GPU util: {np.mean(serial_stats['gpu']):.2f}%")
            logger.log(f"  Parallel GPU util: {np.mean(parallel_stats['gpu']):.2f}%")
            logger.log(f"  Serial memory util: {np.mean(serial_stats['mem']):.2f}%")
            logger.log(f"  Parallel memory util: {np.mean(parallel_stats['mem']):.2f}%")
            logger.log(f"  Serial memory used: {np.mean(serial_stats['mem_used']):.2f} MB")
            logger.log(f"  Parallel memory used: {np.mean(parallel_stats['mem_used']):.2f} MB")
    
    # 程序结束时生成并输出汇总表格
    logger.log("\nSummary Table for All Batch Sizes:")
    print_summary_table(results, logger, args.log_file)  # 传递log_file路径
    
    # 生成结果图表（修复：添加函数调用）
    if args.generate_plot:
        plot_path = args.log_file.replace(".log", "_batch_plot.png")
        generate_batch_size_plot(results, plot_path)
        print(f"Batch size performance plot generated at: {plot_path}")

# 更新__main__块 - 只创建一个logger
if __name__ == "__main__":
    args = parse_args()
    logger = FileLogger(args.log_file)  # 只在这里创建logger
    
    try:
        device = torch.device(f"cuda:{args.cuda_visible_devices}" if torch.cuda.is_available() else "cpu")
        main(args)  # 传递args给main()
    finally:
        logger.close()  # 确保日志关闭
