'''
python example/hf_mistral_dual_stream.py   --cuda_visible_devices 1 \
    --dtype float16   --max_new_tokens 32   --temperature 0.0   --top_p 1.0 \
    --repeats 100 --batch_size 8 --with_gpu_stats \
    --log_file example/hf_mistral_dual_stream.log
'''
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import sys
import time
from datetime import datetime
from typing import List, Tuple
import threading
import subprocess

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class FileLogger:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def write(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 HF Transformers 对比两个批处理的串行 vs 双 stream 并行 vs 单大批处理"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace 模型 ID 或本地路径",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="每个请求生成的最大新 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度 (0 表示贪心)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="核采样 top-p",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="模型精度",
    )
    parser.add_argument(
        "--prompt_a",
        type=str,
        default="You are a helpful assistant. Summarize the importance of unit tests in software engineering in 3 bullet points.",
        help="请求 A 的提示词",
    )
    parser.add_argument(
        "--prompt_b",
        type=str,
        default="Explain, in simple terms, how transformers process text and why attention is useful.",
        help="请求 B 的提示词",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="1",
        help="设置 CUDA_VISIBLE_DEVICES，例如 '1' 只使用第 1 号 GPU",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="重复执行次数",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "hf_mistral_dual_stream.log"),
        help="日志文件路径",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="每个批次的大小（总任务为 2 * batch_size）",
    )
    parser.add_argument(
        "--with_gpu_stats",
        action="store_true",
        help="启用 GPU 利用率和显存统计（使用 nvidia-smi）",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def load_model_and_tokenizer(model_id: str, dtype_str: str, device: torch.device):
    torch_dtype = get_torch_dtype(dtype_str)
    tokenizer = AutoTokenizer.from_pretrained(model_id) # , trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
        # trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_ids(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=model.config.eos_token_id,
            eos_token_id=model.config.eos_token_id,
        )
    return outputs


def _thread_target_stream(model, tensors: Tuple[torch.Tensor, torch.Tensor], stream: torch.cuda.Stream, max_new_tokens: int, temperature: float, top_p: float, error_box: List[Exception]):
    try:
        input_ids, attention_mask = tensors
        with torch.cuda.stream(stream):
            _ = generate_ids(model, input_ids, attention_mask, max_new_tokens, temperature, top_p)
    except Exception as e:
        error_box.append(e)


def benchmark_parallel_streams(model, tensors_pair: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], device: torch.device, max_new_tokens: int, temperature: float, top_p: float) -> float:
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    error_box: List[Exception] = []
    thread_a = threading.Thread(
        target=_thread_target_stream,
        args=(model, tensors_pair[0], stream_a, max_new_tokens, temperature, top_p, error_box),
        daemon=True,
    )
    thread_b = threading.Thread(
        target=_thread_target_stream,
        args=(model, tensors_pair[1], stream_b, max_new_tokens, temperature, top_p, error_box),
        daemon=True,
    )

    start = time.perf_counter()
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()
    torch.cuda.synchronize(device)
    end = time.perf_counter()

    if error_box:
        raise error_box[0]

    return end - start


def benchmark_serial(model, tensors_pair: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], device: torch.device, max_new_tokens: int, temperature: float, top_p: float) -> float:
    (ids_a, mask_a), (ids_b, mask_b) = tensors_pair
    start = time.perf_counter()
    _ = generate_ids(model, ids_a, mask_a, max_new_tokens, temperature, top_p)
    _ = generate_ids(model, ids_b, mask_b, max_new_tokens, temperature, top_p)
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    return end - start


def benchmark_big_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device, max_new_tokens: int, temperature: float, top_p: float) -> float:
    start = time.perf_counter()
    _ = generate_ids(model, input_ids, attention_mask, max_new_tokens, temperature, top_p)
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    return end - start


def collect_gpu_stats(device_index: int = 1) -> tuple[float, float, float]:
    try:
        output = subprocess.check_output(
            f"nvidia-smi -i {device_index} -q -d UTILIZATION,MEMORY",
            shell=True,
            text=True
        )
        lines = output.splitlines()
        util = mem_used = mem_total = 0.0
        in_fb_memory = False
        in_util = False

        for line in lines:
            stripped = line.strip()

            # 进入 FB Memory Usage 部分
            if stripped.startswith("FB Memory Usage"):
                in_fb_memory = True
                continue
            if in_fb_memory and stripped.startswith("Total"):
                mem_total = float(stripped.split(":")[1].strip().split()[0])
            elif in_fb_memory and stripped.startswith("Used"):
                mem_used = float(stripped.split(":")[1].strip().split()[0])
                in_fb_memory = False  # 拿到Used就可以退出这个block

            # 进入 Utilization 部分
            if stripped.startswith("Utilization"):
                in_util = True
                continue
            if in_util and stripped.startswith("Gpu"):
                util = float(stripped.split(":")[1].strip().split("%")[0])
                in_util = False  # GPU拿到就够了

        mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0.0
        return util, mem_util, mem_used
    except Exception as e:
        print(f"GPU stats collection failed: {e}")
        return 0.0, 0.0, 0.0


def main():
    # 固定 GPU1 与稳定性设置
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 降低 CPU/GIL 竞争
    torch.set_num_threads(1)

    args = parse_args()
    logger = FileLogger(args.log_file)

    if not torch.cuda.is_available():
        print("未检测到可用的 CUDA。")
        sys.exit(1)

    device = torch.device("cuda:0")  # 映射到可见 GPU1 的第 0 号

    print("[HF] 正在加载模型至 GPU1……")
    logger.write(f"Loading HF model: {args.model}, dtype={args.dtype}, device={device}, batch_size={args.batch_size}")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, device)

    # 准备两个批次的提示（每个批次交替 A 和 B 填充到 batch_size）
    prompts_batch1 = [args.prompt_a if i % 2 == 0 else args.prompt_b for i in range(args.batch_size)]
    prompts_batch2 = [args.prompt_b if i % 2 == 0 else args.prompt_a for i in range(args.batch_size)]  # 略作不同以平衡

    # 批处理分词与 padding
    enc_batch1 = tokenizer(prompts_batch1, return_tensors="pt", padding=True)
    input_ids_batch1 = enc_batch1["input_ids"].to(device, non_blocking=True)
    attention_mask_batch1 = enc_batch1["attention_mask"].to(device, non_blocking=True)

    enc_batch2 = tokenizer(prompts_batch2, return_tensors="pt", padding=True)
    input_ids_batch2 = enc_batch2["input_ids"].to(device, non_blocking=True)
    attention_mask_batch2 = enc_batch2["attention_mask"].to(device, non_blocking=True)

    tensors_pair = ((input_ids_batch1, attention_mask_batch1), (input_ids_batch2, attention_mask_batch2))

    # 拼接成大 batch（需统一 padding 到全局 max length）
    all_prompts = prompts_batch1 + prompts_batch2
    enc_big = tokenizer(all_prompts, return_tensors="pt", padding=True)
    input_ids_big = enc_big["input_ids"].to(device, non_blocking=True)
    attention_mask_big = enc_big["attention_mask"].to(device, non_blocking=True)

    # 预热（串行两个批处理 + 双 stream 两个批处理 + 单大 batch）
    logger.write("Running warmup (serial & streams & big batch)...")
    _ = benchmark_serial(model, tensors_pair, device, max_new_tokens=8, temperature=0.0, top_p=1.0)  # 串行预热
    _ = benchmark_parallel_streams(model, tensors_pair, device, max_new_tokens=8, temperature=0.0, top_p=1.0)  # 双 stream 预热
    _ = benchmark_big_batch(model, input_ids_big[:4], attention_mask_big[:4], device, max_new_tokens=8, temperature=0.0, top_p=1.0)  # 大 batch 预热 (小部分)

    # 串行基准
    print(f"[HF] 开始串行两个批处理（每个 {args.batch_size} 请求）计时，重复 {args.repeats} 次……")
    serial_gpu_utils = []
    serial_mem_utils = []
    serial_mem_useds = []
    t0 = time.perf_counter()
    for _ in tqdm(range(args.repeats), desc="Serial Progress"):
        if args.with_gpu_stats:
            util, mem_util, mem_used = collect_gpu_stats()
            serial_gpu_utils.append(util)
            serial_mem_utils.append(mem_util)
            serial_mem_useds.append(mem_used)
        _ = benchmark_serial(model, tensors_pair, device, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    t1 = time.perf_counter()
    serial_total_s = t1 - t0
    total_requests = args.repeats * args.batch_size * 2
    serial_avg_task_s = serial_total_s / total_requests
    serial_throughput = total_requests / serial_total_s if serial_total_s > 0 else 0.0
    serial_avg_gpu_util = sum(serial_gpu_utils) / len(serial_gpu_utils) if serial_gpu_utils else 0.0
    serial_avg_mem_util = sum(serial_mem_utils) / len(serial_mem_utils) if serial_mem_utils else 0.0
    serial_avg_mem_used = sum(serial_mem_useds) / len(serial_mem_useds) if serial_mem_useds else 0.0

    # 双 stream 并行基准
    print(f"[HF] 开始双 stream 并行两个批处理（每个 {args.batch_size} 请求）计时，重复 {args.repeats} 次……")
    streams_gpu_utils = []
    streams_mem_utils = []
    streams_mem_useds = []
    t0 = time.perf_counter()
    for _ in tqdm(range(args.repeats), desc="Streams Progress"):
        if args.with_gpu_stats:
            util, mem_util, mem_used = collect_gpu_stats()
            streams_gpu_utils.append(util)
            streams_mem_utils.append(mem_util)
            streams_mem_useds.append(mem_used)
        _ = benchmark_parallel_streams(model, tensors_pair, device, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    t1 = time.perf_counter()
    streams_total_s = t1 - t0
    streams_avg_task_s = streams_total_s / total_requests
    streams_throughput = total_requests / streams_total_s if streams_total_s > 0 else 0.0
    streams_avg_gpu_util = sum(streams_gpu_utils) / len(streams_gpu_utils) if streams_gpu_utils else 0.0
    streams_avg_mem_util = sum(streams_mem_utils) / len(streams_mem_utils) if streams_mem_utils else 0.0
    streams_avg_mem_used = sum(streams_mem_useds) / len(streams_mem_useds) if streams_mem_useds else 0.0

    # 单大 batch 基准
    print(f"[HF] 开始单大批处理（2*{args.batch_size} 请求）计时，重复 {args.repeats} 次……")
    big_gpu_utils = []
    big_mem_utils = []
    big_mem_useds = []
    t0 = time.perf_counter()
    for _ in tqdm(range(args.repeats), desc="Big Batch Progress"):
        if args.with_gpu_stats:
            util, mem_util, mem_used = collect_gpu_stats()
            big_gpu_utils.append(util)
            big_mem_utils.append(mem_util)
            big_mem_useds.append(mem_used)
        _ = benchmark_big_batch(model, input_ids_big, attention_mask_big, device, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    t1 = time.perf_counter()
    big_total_s = t1 - t0
    big_avg_task_s = big_total_s / total_requests
    big_throughput = total_requests / big_total_s if big_total_s > 0 else 0.0
    big_avg_gpu_util = sum(big_gpu_utils) / len(big_gpu_utils) if big_gpu_utils else 0.0
    big_avg_mem_util = sum(big_mem_utils) / len(big_mem_utils) if big_mem_utils else 0.0
    big_avg_mem_used = sum(big_mem_useds) / len(big_mem_useds) if big_mem_useds else 0.0

    # 对比（串行 vs 双 stream vs 单大 batch）
    comparison = (
        f"Serial: total={serial_total_s:.4f}s, avg_task={serial_avg_task_s:.4f}s, throughput={serial_throughput:.2f} req/s"
    )
    if args.with_gpu_stats:
        comparison += f", gpu_util={serial_avg_gpu_util:.1f}%, mem_util={serial_avg_mem_util:.1f}%, mem_used={serial_avg_mem_used:.1f} MiB"
    comparison += (
        f" | Streams: total={streams_total_s:.4f}s, avg_task={streams_avg_task_s:.4f}s, throughput={streams_throughput:.2f} req/s"
    )
    if args.with_gpu_stats:
        comparison += f", gpu_util={streams_avg_gpu_util:.1f}%, mem_util={streams_avg_mem_util:.1f}%, mem_used={streams_avg_mem_used:.1f} MiB"
    comparison += (
        f" | BigBatch: total={big_total_s:.4f}s, avg_task={big_avg_task_s:.4f}s, throughput={big_throughput:.2f} req/s"
    )
    if args.with_gpu_stats:
        comparison += f", gpu_util={big_avg_gpu_util:.1f}%, mem_util={big_avg_mem_util:.1f}%, mem_used={big_avg_mem_used:.1f} MiB"
    comparison += f" | Streams Speedup vs Serial: {(serial_total_s / streams_total_s):.2f}x | BigBatch Speedup vs Serial: {(serial_total_s / big_total_s):.2f}x"

    logger.write(f"Serial total time for {args.repeats} runs (2 batches/run, {args.batch_size} req/batch): {serial_total_s:.4f} s")
    logger.write(f"Serial average per request: {serial_avg_task_s:.4f} s")
    logger.write(f"Serial throughput: {serial_throughput:.2f} req/s")
    if args.with_gpu_stats:
        logger.write(f"Serial avg GPU util: {serial_avg_gpu_util:.1f}%")
        logger.write(f"Serial avg mem util: {serial_avg_mem_util:.1f}%")
        logger.write(f"Serial avg mem used: {serial_avg_mem_used:.1f} MiB")
    logger.write(f"Streams total time for {args.repeats} runs (2 batches/run, {args.batch_size} req/batch): {streams_total_s:.4f} s")
    logger.write(f"Streams average per request: {streams_avg_task_s:.4f} s")
    logger.write(f"Streams throughput: {streams_throughput:.2f} req/s")
    if args.with_gpu_stats:
        logger.write(f"Streams avg GPU util: {streams_avg_gpu_util:.1f}%")
        logger.write(f"Streams avg mem util: {streams_avg_mem_util:.1f}%")
        logger.write(f"Streams avg mem used: {streams_avg_mem_used:.1f} MiB")
    logger.write(f"BigBatch total time for {args.repeats} runs (1 batch/run, 2*{args.batch_size} req/batch): {big_total_s:.4f} s")
    logger.write(f"BigBatch average per request: {big_avg_task_s:.4f} s")
    logger.write(f"BigBatch throughput: {big_throughput:.2f} req/s")
    if args.with_gpu_stats:
        logger.write(f"BigBatch avg GPU util: {big_avg_gpu_util:.1f}%")
        logger.write(f"BigBatch avg mem util: {big_avg_mem_util:.1f}%")
        logger.write(f"BigBatch avg mem used: {big_avg_mem_used:.1f} MiB")
    logger.write(comparison)

    print("[HF] 对比结果：")
    print(comparison)
    print(f"[HF] 详细日志已写入: {args.log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断。")
        sys.exit(130)
    except Exception as exc:
        print(f"运行出错: {exc}")
        sys.exit(1) 