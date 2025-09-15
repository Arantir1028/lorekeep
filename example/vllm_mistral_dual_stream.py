import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Tuple
import threading


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
        description="使用 vLLM 在两个 CUDA stream 上并行执行两次生成，并测量端到端时间"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace 模型 ID 或本地路径",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="每个请求生成的最大新 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="核采样 top-p",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="张量并行度",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="模型精度",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "vllm_mistral_dual_stream.log"),
        help="日志文件路径",
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
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="vLLM 可使用的显存比例(0-1)，适当降低以避免 OOM",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="限制最大上下文长度以减少 KV Cache 占用",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=2,
        help="并发序列上限，设为 2 以适配双请求",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=0,
        help="调度批处理的 token 上限，0 表示自动按 max_num_seqs*max_model_len 设置",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="启用 eager 前端，避免编译器引入的额外峰值显存",
    )
    parser.add_argument(
        "--concurrency_mode",
        type=str,
        choices=["batched", "threads"],
        default="batched",
        help="选择并发方式: batched(推荐) 或 threads(可能导致阻塞)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="每种方式重复执行的次数，用于统计总时间和平均时间",
    )
    return parser.parse_args()


def resolve_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    import torch
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def get_llm(model: str, tensor_parallel_size: int, dtype_str: str, gpu_memory_utilization: float, max_model_len: int, max_num_seqs: int, enforce_eager_flag: bool, max_num_batched_tokens: int):
    from vllm import LLM

    dtype = resolve_dtype(dtype_str)
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True,
        enforce_eager=enforce_eager_flag,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens if max_num_batched_tokens > 0 else None,
    )
    return llm


def build_sampling_params(max_tokens: int, temperature: float, top_p: float):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=None,
    )


def run_generate(llm, prompts: List[str], sampling_params) -> List[str]:
    outputs = llm.generate(prompts, sampling_params)
    texts: List[str] = []
    for output in outputs:
        texts.append(output.outputs[0].text)
    return texts


def _thread_target_with_stream(llm, prompt: str, sampling_params, stream, error_box: List[Exception]):
    try:
        import torch
        if torch.cuda.is_available() and stream is not None:
            with torch.cuda.stream(stream):
                run_generate(llm, [prompt], sampling_params)
                torch.cuda.synchronize()
        else:
            run_generate(llm, [prompt], sampling_params)
    except Exception as e:
        error_box.append(e)


def benchmark_dual_stream(llm, prompts_pair: Tuple[str, str], sampling_params) -> float:
    import torch

    use_cuda_streams = torch.cuda.is_available()
    stream_a = torch.cuda.Stream() if use_cuda_streams else None
    stream_b = torch.cuda.Stream() if use_cuda_streams else None

    error_box: List[Exception] = []
    thread_a = threading.Thread(
        target=_thread_target_with_stream,
        args=(llm, prompts_pair[0], sampling_params, stream_a, error_box),
        daemon=True,
    )
    thread_b = threading.Thread(
        target=_thread_target_with_stream,
        args=(llm, prompts_pair[1], sampling_params, stream_b, error_box),
        daemon=True,
    )

    start = time.perf_counter()
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()
    end = time.perf_counter()

    if error_box:
        raise error_box[0]

    return end - start


def benchmark_batched(llm, prompts_pair: Tuple[str, str], sampling_params) -> float:
    start = time.perf_counter()
    run_generate(llm, [prompts_pair[0], prompts_pair[1]], sampling_params)
    end = time.perf_counter()
    return end - start


def benchmark_serial(llm, prompts_pair: Tuple[str, str], sampling_params) -> float:
    start = time.perf_counter()
    run_generate(llm, [prompts_pair[0]], sampling_params)
    run_generate(llm, [prompts_pair[1]], sampling_params)
    end = time.perf_counter()
    return end - start


def main():
    # 设备与稳定性环境变量
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("VLLM_TOKENIZER_MODE", "mistral")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()

    # 固定使用指定 GPU（默认 1 号），在导入/初始化 CUDA 模块前生效
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # 规范 max_num_batched_tokens，避免调度告警
    if args.max_num_batched_tokens <= 0:
        args.max_num_batched_tokens = args.max_num_seqs * args.max_model_len

    logger = FileLogger(args.log_file)

    print(f"[vLLM] 使用 GPU(s): {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}，正在加载模型……")
    logger.write(
        f"Loading model: {args.model}, tp={args.tensor_parallel_size}, dtype={args.dtype}, "
        f"gpu_mem_util={args.gpu_memory_utilization}, max_model_len={args.max_model_len}, "
        f"max_num_seqs={args.max_num_seqs}, enforce_eager={args.enforce_eager}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}, concurrency_mode={args.concurrency_mode}, repeats={args.repeats}"
    )

    llm = get_llm(
        args.model,
        args.tensor_parallel_size,
        args.dtype,
        args.gpu_memory_utilization,
        args.max_model_len,
        args.max_num_seqs,
        args.enforce_eager,
        args.max_num_batched_tokens,
    )

    sampling_params = build_sampling_params(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    logger.write("Running warmup...")
    try:
        run_generate(llm, ["Warmup: say hello."], sampling_params)
    except Exception as e:
        logger.write(f"Warmup failed: {e}")
        raise

    # 先运行并发模式，重复执行
    if args.concurrency_mode == "batched":
        print(f"[vLLM] 开始批处理两请求（由引擎并发执行）计时，重复 {args.repeats} 次……")
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            benchmark_batched(llm, (args.prompt_a, args.prompt_b), sampling_params)
        t1 = time.perf_counter()
        parallel_total_s = t1 - t0
        parallel_avg_task_s = parallel_total_s / (args.repeats * 2)
    else:
        print(f"[vLLM] 开始双线程 + 双 CUDA stream 并行计时（可能阻塞），重复 {args.repeats} 次……")
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            benchmark_dual_stream(llm, (args.prompt_a, args.prompt_b), sampling_params)
        t1 = time.perf_counter()
        parallel_total_s = t1 - t0
        parallel_avg_task_s = parallel_total_s / (args.repeats * 2)

    # 再运行串行基准，重复执行
    print(f"[vLLM] 开始串行两次推理计时，重复 {args.repeats} 次……")
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        benchmark_serial(llm, (args.prompt_a, args.prompt_b), sampling_params)
    t1 = time.perf_counter()
    serial_total_s = t1 - t0
    serial_avg_task_s = serial_total_s / (args.repeats * 2)

    # 对比
    speedup = serial_total_s / parallel_total_s if parallel_total_s > 0 else float("inf")
    comparison = (
        f"Serial: total={serial_total_s:.4f}s, avg_task={serial_avg_task_s:.4f}s | "
        f"Parallel({args.concurrency_mode}): total={parallel_total_s:.4f}s, avg_task={parallel_avg_task_s:.4f}s | "
        f"Speedup: {speedup:.2f}x"
    )

    logger.write(f"Serial total time for {args.repeats} runs (2 tasks/run): {serial_total_s:.4f} s")
    logger.write(f"Serial average per task: {serial_avg_task_s:.4f} s")
    logger.write(f"Parallel total time for {args.repeats} runs (2 tasks/run, {args.concurrency_mode}): {parallel_total_s:.4f} s")
    logger.write(f"Parallel average per task: {parallel_avg_task_s:.4f} s")
    logger.write(comparison)

    print("[vLLM] 对比结果：")
    print(comparison)
    print(f"[vLLM] 详细日志已写入: {args.log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断。")
        sys.exit(130)
    except Exception as exc:
        print(f"运行出错: {exc}")
        sys.exit(1) 