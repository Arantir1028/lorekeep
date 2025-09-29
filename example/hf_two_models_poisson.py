'''
# 请求到达时间符合泊松分布，两个模型，并行执行
python example/hf_two_models_poisson.py --cuda_visible_devices 0 --max_new_tokens 128 --total_requests 512 --lambda_rate 2 --batch_size 0 --with_gpu_stats --log_file example/hf_two_models_poisson.log --generate_plot
'''
import os
import time
import torch
import threading
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import csv
import logging
from collections import defaultdict
import heapq  # 用于模拟时间优先队列

# 日志设置
class Logger:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log(self, message):
        self.logger.info(message)

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
        logging.shutdown()

logger = None

# 加载模型和tokenizer
def load_model_and_tokenizer(model_name, args, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    model.eval()
    return model, tokenizer

# 手动生成函数（保留原有，用于真stream并行）
def manual_generate(model, input_ids, attention_mask, max_new_tokens, device, stream=None, chunk_size=4):
    if stream is not None:
        torch.cuda.set_stream(stream)
    seq_length = input_ids.shape[1]
    past_key_values = None
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens // chunk_size):
        with torch.no_grad():
            for __ in range(chunk_size):
                outputs = model(input_ids=generated_ids[:, -1:], attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                generated_ids = torch.cat((generated_ids, next_token), dim=1)
                attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 1), device=device)), dim=1)
                past_key_values = outputs.past_key_values
    if stream is not None:
        stream.synchronize()
    return generated_ids

# GPU监控线程
def gpu_monitor(stop_event, gpu_utils, mem_utils, mem_used, device_index=0, device=None, sample_interval=0.05):
    while not stop_event.is_set():
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "-i", str(device_index), "--query-gpu=utilization.gpu,utilization.memory,memory.used", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            ).strip().splitlines()[0]  # 取第一行（针对多GPU）
            util_gpu, util_mem, mem_u = map(float, result.split(","))
            gpu_utils.append(util_gpu)
            mem_utils.append(util_mem)
            mem_used.append(mem_u)  # 单位 MiB
            # Cross-check with torch
            torch_mem = torch.cuda.memory_allocated(device) / (1024 ** 2) if device else 0  # MiB
        except Exception as e:
            logger.log(f"GPU stats collection failed: {e}")
        time.sleep(sample_interval)

# 获取物理GPU索引（根据用户反馈，0->GPU1, 1->GPU0）
def get_physical_gpu_index(logical_index):
    return 1 - logical_index

# 基准测试：serial模式
def benchmark_serial(models, tokenizers, input_ids_list, attention_mask_list, args, device, gpu_utils=None, mem_utils=None, mem_used=None):
    start_time = time.time()
    model_times = []
    cum_offset = 0.0
    for m_idx, model in enumerate(models):
        if input_ids_list[m_idx] is None:
            model_times.append(0.0)
            continue
        input_ids = input_ids_list[m_idx].to(device)
        attention_mask = attention_mask_list[m_idx].to(device)
        start_e = torch.cuda.Event(enable_timing=True)
        end_e = torch.cuda.Event(enable_timing=True)
        start_e.record()
        _ = manual_generate(model, input_ids, attention_mask, args.max_new_tokens, device)
        end_e.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start_e.elapsed_time(end_e)
        duration = elapsed_ms / 1000.0
        model_times.append(duration)
        cum_offset += duration
    total_time = cum_offset  # 使用累积GPU时间作为total
    return total_time, model_times

# 基准测试：parallel模式
def benchmark_parallel_threads(models, tokenizers, input_ids_list, attention_mask_list, args, device, gpu_utils=None, mem_utils=None, mem_used=None):
    start_time = time.time()
    model_times = [0.0] * len(models)
    streams = [torch.cuda.Stream(device=device) for _ in models]

    def model_thread(m_idx, model, input_ids, attention_mask, max_new_tokens, device, stream, timing_list):
        if input_ids is None:
            timing_list[m_idx] = 0.0
            return
        with torch.cuda.stream(stream):
            start_e = torch.cuda.Event(enable_timing=True)
            end_e = torch.cuda.Event(enable_timing=True)
            start_e.record()
            _ = manual_generate(model, input_ids, attention_mask, max_new_tokens, device, stream)
            end_e.record()
            stream.synchronize()   # 正确同步流
            elapsed_ms = start_e.elapsed_time(end_e)
            timing_list[m_idx] = elapsed_ms / 1000.0

    batch_start_time = time.time()
    threads = []
    for m_idx, model in enumerate(models):
        input_ids = input_ids_list[m_idx].to(device) if input_ids_list[m_idx] is not None else None
        attention_mask = attention_mask_list[m_idx].to(device) if attention_mask_list[m_idx] is not None else None
        t = threading.Thread(target=model_thread, args=(m_idx, model, input_ids, attention_mask, args.max_new_tokens, device, streams[m_idx], model_times))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize(device)
    total_time = max(model_times) if model_times else 0
    return total_time, model_times

# 新：泊松模拟函数
def simulate_poisson_arrivals(lambda_rate, total_requests):
    arrival_times = []
    current_time = 0
    while len(arrival_times) < total_requests:
        inter_arrival = np.random.exponential(1 / lambda_rate)
        current_time += inter_arrival
        arrival_times.append(current_time)
    return arrival_times

# 新：动态批处理模拟
def simulate_dynamic_batching(mode, models, tokenizers, prompts, args, device, lambda_rate, total_requests, gpu_utils, mem_utils, mem_used):
    arrival_times = simulate_poisson_arrivals(lambda_rate, total_requests)
    request_queue = []  # (arrival_time, prompt_idx, model_idx)
    completion_times = []  # 每个请求的响应时间
    sim_time = 0.0
    request_id = 0
    last_completion_time = 0.0  # 跟踪最后一个完成时间

    num_prompts = len(prompts)
    while request_id < total_requests or request_queue:  # 继续直到所有请求处理完
        # 添加新到达请求
        while request_id < total_requests and arrival_times[request_id] <= sim_time:
            model_idx = request_id % 2
            prompt_idx = request_id % num_prompts
            heapq.heappush(request_queue, (arrival_times[request_id], prompt_idx, model_idx))
            request_id += 1

        current_batch_size = len(request_queue)
        if current_batch_size > 0:
            to_pop = current_batch_size if request_id >= total_requests else min(current_batch_size, args.batch_size)
            if to_pop > 0:
                batch = [heapq.heappop(request_queue) for _ in range(to_pop)]
            batch_start = time.time()  # 实际系统开始时间

            prompts_per_model = [[], []]
            batch_arrival_times_per_model = [[], []]  # 每个模型的arr_times
            for arr_time, p_idx, m_idx in batch:
                prompts_per_model[m_idx].append(prompts[p_idx])
                batch_arrival_times_per_model[m_idx].append(arr_time)

            input_ids_list = [None, None]
            attention_mask_list = [None, None]
            has_input = False
            for m_idx in range(2):
                if prompts_per_model[m_idx]:
                    inputs = tokenizers[m_idx](prompts_per_model[m_idx], return_tensors="pt", padding=True)
                    input_ids_list[m_idx] = inputs.input_ids.to(device)
                    attention_mask_list[m_idx] = inputs.attention_mask.to(device)
                    has_input = True

            if not has_input:
                sim_time += 0.01
                continue

            # 执行并获取实际处理时间
            if mode == "serial":
                actual_duration, model_times = benchmark_serial(models, tokenizers, input_ids_list, attention_mask_list, args, device, gpu_utils, mem_utils, mem_used)
            elif mode == "parallel":
                actual_duration, model_times = benchmark_parallel_threads(models, tokenizers, input_ids_list, attention_mask_list, args, device, gpu_utils, mem_utils, mem_used)

            # 推进模拟时间
            batch_start_sim = sim_time
            sim_time += actual_duration
            last_completion_time = sim_time

            # 计算每个请求的响应时间（等待 + 处理）
            if mode == "serial":
                cum_offset = 0.0
                for m_idx in range(2):
                    dur = model_times[m_idx]
                    for arr_time in batch_arrival_times_per_model[m_idx]:
                        wait_time = batch_start_sim - arr_time
                        completion_times.append(wait_time + cum_offset + dur)
                    cum_offset += dur
            elif mode == "parallel":
                for m_idx in range(2):
                    dur = model_times[m_idx]
                    for arr_time in batch_arrival_times_per_model[m_idx]:
                        wait_time = batch_start_sim - arr_time
                        completion_times.append(wait_time + dur)

        else:
            if request_id < total_requests:
                sim_time = max(sim_time, arrival_times[request_id])
            else:
                sim_time += 0.01

    end_to_end_time = last_completion_time - arrival_times[0] if arrival_times else 0
    avg_request_time = np.mean(completion_times) if completion_times else 0
    throughput = total_requests / end_to_end_time if end_to_end_time > 0 else 0
    return end_to_end_time, avg_request_time, throughput

# 绘图函数（保留原有，适应新指标）
def generate_batch_size_plot(results, output_path):
    batch_sizes = results['batch_sizes']
    metrics = ['end_to_end_times', 'avg_request_times', 'throughputs', 'gpu_utils_avg', 'mem_utils_avg', 'mem_used_avg']
    colors = {'serial': 'skyblue', 'parallel': 'orange'}

    for metric in metrics:
        fig, ax = plt.subplots()
        for mode in ['serial', 'parallel']:
            means = [v[0] for v in results[metric][mode]]
            stds = [v[1] for v in results[metric][mode]]
            if len(means) != len(batch_sizes):
                logger.log(f"Warning: Skipping plot for {metric} due to size mismatch")
                continue
            ax.errorbar(batch_sizes, means, yerr=stds, label=mode.capitalize(), color=colors[mode], marker='o', capsize=5)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xscale('log', base=2)
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels(batch_sizes)
        ax.legend()
        plt.savefig(output_path.replace('.png', f'_{metric}.png'))
        plt.close()

# 打印汇总表格（适应新指标）
def print_summary_table(results, logger, log_file):
    headers = ['Batch Size', 'Mode', 'End-to-End Time (s)', 'Avg Request Time (s)', 'Throughput (req/s)', 'Avg GPU Util (%)', 'Avg Mem Util (%)', 'Avg Mem Used (MiB)']
    table = []
    for i, bs in enumerate(results['batch_sizes']):
        for mode in ['serial', 'parallel']:
            row = [bs, mode,
                   f"{results['end_to_end_times'][mode][i][0]:.2f} ± {results['end_to_end_times'][mode][i][1]:.2f}",
                   f"{results['avg_request_times'][mode][i][0]:.2f} ± {results['avg_request_times'][mode][i][1]:.2f}",
                   f"{results['throughputs'][mode][i][0]:.2f} ± {results['throughputs'][mode][i][1]:.2f}",
                   f"{results['gpu_utils_avg'][mode][i][0]:.2f} ± {results['gpu_utils_avg'][mode][i][1]:.2f}",
                   f"{results['mem_utils_avg'][mode][i][0]:.2f} ± {results['mem_utils_avg'][mode][i][1]:.2f}",
                   f"{results['mem_used_avg'][mode][i][0]:.2f} ± {results['mem_used_avg'][mode][i][1]:.2f}"]
            table.append(row)
    logger.log('\t'.join(headers))
    for row in table:
        logger.log('\t'.join(map(str, row)))

    # 输出CSV（包含均值和std）
    csv_path = log_file.replace('.log', '.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers + ['Std End-to-End', 'Std Avg Request', 'Std Throughput', 'Std GPU Util', 'Std Mem Util', 'Std Mem Used'])
        for i, bs in enumerate(results['batch_sizes']):
            for mode in ['serial', 'parallel']:
                mean_row = [bs, mode] + [m[0] for m in [results[k][mode][i] for k in ['end_to_end_times', 'avg_request_times', 'throughputs', 'gpu_utils_avg', 'mem_utils_avg', 'mem_used_avg']]]
                std_row = [m[1] for m in [results[k][mode][i] for k in ['end_to_end_times', 'avg_request_times', 'throughputs', 'gpu_utils_avg', 'mem_utils_avg', 'mem_used_avg']]]
                writer.writerow(mean_row + std_row)
    logger.log(f"Results saved to CSV: {csv_path}")

# 主函数
def main(args):
    global logger
    logger = Logger(args.log_file)
    device = torch.device(f"cuda:{args.cuda_visible_devices}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # 加载模型
    model_names = ["mistralai/Mistral-7B-Instruct-v0.2", "tiiuae/falcon-7b-instruct"]
    models = []
    tokenizers = []
    for model_name in model_names:
        model, tokenizer = load_model_and_tokenizer(model_name, args, device)
        models.append(model)
        tokenizers.append(tokenizer)

    # 示例prompts（可扩展）
    prompts = ["Tell me a joke.", "What is AI?", "Explain quantum computing.", "Write a short story."] * 250  # 足够多以循环使用

    # 测试batch_sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128] if args.batch_size == 0 else [args.batch_size]

    # 结果存储（为多次试验准备）
    results = defaultdict(lambda: defaultdict(list))  # {metric: {mode: [values]}}

    num_trials = args.num_trials
    results['batch_sizes'] = batch_sizes
    for batch_size in tqdm(batch_sizes, desc="Testing batch sizes"):
        args.batch_size = batch_size
        trial_end_to_end = {'serial': [], 'parallel': []}
        trial_avg_request = {'serial': [], 'parallel': []}
        trial_throughput = {'serial': [], 'parallel': []}
        trial_gpu_avg = {'serial': [], 'parallel': []}
        trial_mem_avg = {'serial': [], 'parallel': []}
        trial_mem_used_avg = {'serial': [], 'parallel': []}
        for trial in range(num_trials):
            for mode in ['serial', 'parallel']:
                gpu_utils, mem_utils, mem_used = [], [], []
                stop_event = threading.Event()
                if args.with_gpu_stats:
                    physical_index = get_physical_gpu_index(int(args.cuda_visible_devices))
                    monitor_thread = threading.Thread(target=gpu_monitor, args=(stop_event, gpu_utils, mem_utils, mem_used, physical_index, device))
                    monitor_thread.start()

                end_to_end_time, avg_request_time, throughput = simulate_dynamic_batching(mode, models, tokenizers, prompts, args, device, args.lambda_rate, args.total_requests, gpu_utils, mem_utils, mem_used)

                if args.with_gpu_stats:
                    stop_event.set()
                    monitor_thread.join()

                trial_end_to_end[mode].append(end_to_end_time)
                trial_avg_request[mode].append(avg_request_time)
                trial_throughput[mode].append(throughput)
                trial_gpu_avg[mode].append(np.mean(gpu_utils) if gpu_utils else 0)
                trial_mem_avg[mode].append(np.mean(mem_utils) if mem_utils else 0)
                trial_mem_used_avg[mode].append(np.mean(mem_used) if mem_used else 0)

        for mode in ['serial', 'parallel']:
            results['end_to_end_times'][mode].append((np.mean(trial_end_to_end[mode]), np.std(trial_end_to_end[mode])))
            results['avg_request_times'][mode].append((np.mean(trial_avg_request[mode]), np.std(trial_avg_request[mode])))
            results['throughputs'][mode].append((np.mean(trial_throughput[mode]), np.std(trial_throughput[mode])))
            results['gpu_utils_avg'][mode].append((np.mean(trial_gpu_avg[mode]), np.std(trial_gpu_avg[mode])))
            results['mem_utils_avg'][mode].append((np.mean(trial_mem_avg[mode]), np.std(trial_mem_avg[mode])))
            results['mem_used_avg'][mode].append((np.mean(trial_mem_used_avg[mode]), np.std(trial_mem_used_avg[mode])))

    # 输出汇总
    logger.log("\nSummary Table for All Batch Sizes:")
    print_summary_table(results, logger, args.log_file)

    # 生成图表
    if args.generate_plot:
        plot_path = args.log_file.replace(".log", "_batch_plot.png")
        generate_batch_size_plot(results, plot_path)
        logger.log(f"Batch size performance plot generated at: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_visible_devices", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=100)  # 保留，但泊松模拟不直接使用
    parser.add_argument("--batch_size", type=int, default=0)  # 0表示测试所有
    parser.add_argument("--with_gpu_stats", action="store_true")
    parser.add_argument("--log_file", type=str, default="example/hf_two_models_poisson.log")
    parser.add_argument("--generate_plot", action="store_true")
    parser.add_argument("--lambda_rate", type=float, default=1.0)  # 新：泊松到达率
    parser.add_argument("--total_requests", type=int, default=1000)  # 新：总请求数
    parser.add_argument("--num_trials", type=int, default=3)  # 新：试验次数
    args = parser.parse_args()
    main(args)
    if logger:
        logger.close()