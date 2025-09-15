import argparse
import os
import sys
import time
import threading
from datetime import datetime
from typing import List, Tuple

# Prefer minimal console prints; log detailed results to file per user preference
# Logging helper
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
        description="Benchmark vLLM inference: serial vs dual-thread with two CUDA streams"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model id or local path",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Max new tokens to generate per request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallelism degree",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "vllm_mistral_benchmark.log"),
        help="Path to write detailed logs",
    )
    parser.add_argument(
        "--prompt_a",
        type=str,
        default="You are a helpful assistant. Summarize the importance of unit tests in software engineering in 3 bullet points.",
        help="Prompt for request A",
    )
    parser.add_argument(
        "--prompt_b",
        type=str,
        default="Explain, in simple terms, how transformers process text and why attention is useful.",
        help="Prompt for request B",
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


def get_llm(model: str, tensor_parallel_size: int, dtype_str: str):
    from vllm import LLM

    dtype = resolve_dtype(dtype_str)
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=True,
        enforce_eager=False,
        # Keep within a single process (no Ray) and default engine scheduling
        # engine_use_ray=False is default when using LLM API
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
    # Returns generated texts for each prompt (deterministic order)
    outputs = llm.generate(prompts, sampling_params)
    texts: List[str] = []
    for output in outputs:
        # output.outputs is a list of candidates; take the first
        texts.append(output.outputs[0].text)
    return texts


def benchmark_serial(llm, prompts_pair: Tuple[str, str], sampling_params) -> float:
    start = time.perf_counter()
    run_generate(llm, [prompts_pair[0]], sampling_params)
    run_generate(llm, [prompts_pair[1]], sampling_params)
    end = time.perf_counter()
    return end - start


# Keep the thread+stream code for reference, but not used to avoid deadlocks in vLLM front-end
def _thread_target_with_stream(llm, prompt: str, sampling_params, stream, error_box: List[Exception]):
    try:
        import torch
        if torch.cuda.is_available():
            with torch.cuda.stream(stream):
                run_generate(llm, [prompt], sampling_params)
                torch.cuda.synchronize()
        else:
            run_generate(llm, [prompt], sampling_params)
    except Exception as e:
        error_box.append(e)


def benchmark_parallel_batched(llm, prompts_pair: Tuple[str, str], sampling_params) -> float:
    # Submit both requests together to let vLLM's scheduler execute them concurrently on a single GPU
    start = time.perf_counter()
    run_generate(llm, [prompts_pair[0], prompts_pair[1]], sampling_params)
    end = time.perf_counter()
    return end - start


def main():
    # Environment suggestions for stability and correct tokenization behavior
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("VLLM_TOKENIZER_MODE", "mistral")

    args = parse_args()
    logger = FileLogger(args.log_file)

    # Load model
    print("[vLLM] 正在加载模型，请稍候……")
    logger.write(f"Loading model: {args.model}, tp={args.tensor_parallel_size}, dtype={args.dtype}")
    llm = get_llm(args.model, args.tensor_parallel_size, args.dtype)

    # Prepare sampling params
    sampling_params = build_sampling_params(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Warmup run to exclude one-time initialization
    logger.write("Running warmup...")
    try:
        run_generate(llm, ["Warmup: say hello."], sampling_params)
    except Exception as e:
        logger.write(f"Warmup failed: {e}")
        raise

    # Serial benchmark
    print("[vLLM] 开始串行两次推理计时……")
    serial_s = benchmark_serial(llm, (args.prompt_a, args.prompt_b), sampling_params)
    logger.write(f"Serial total E2E time (2 requests): {serial_s:.4f} s")

    # Parallel benchmark using batched submission (engine-managed concurrency on single GPU)
    print("[vLLM] 开始并行（批处理两请求，由引擎并发执行）计时……")
    parallel_s = benchmark_parallel_batched(
        llm, (args.prompt_a, args.prompt_b), sampling_params
    )
    logger.write(f"Parallel total E2E time (2 requests, batched): {parallel_s:.4f} s")

    # Compare and print concise result
    speedup = serial_s / parallel_s if parallel_s > 0 else float("inf")
    comparison = (
        f"Serial: {serial_s:.4f}s | Parallel(Batch=2): {parallel_s:.4f}s | Speedup: {speedup:.2f}x"
    )
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