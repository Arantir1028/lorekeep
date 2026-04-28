# lorekeep/profiler/offline_profiler.py

import argparse
import contextlib
import torch
import torch.nn.functional as F
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ModuleNotFoundError:
    class SDPBackend:
        FLASH_ATTENTION = "flash"

    @contextlib.contextmanager
    def sdpa_kernel(backends=None):
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            ):
                yield
        else:
            yield
import json
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import hw_config as cfg
from tools.experiment_lock import gpu_experiment_lock

# =====================================================================
# 系统顶会级别基准测试常量
# =====================================================================
WARMUP_ITERS = 10
ACTIVE_ITERS = 50

def simulate_attention(Q, K, V, q_heads, kv_heads):
    """
    动态兼容 MHA / GQA / MQA 的底层张量展开逻辑。
    为获取最纯粹的物理访存极限，剥离 is_causal 掩码，强制硬件跑 Dense 矩阵乘。
    """
    num_key_value_groups = q_heads // kv_heads
    if num_key_value_groups > 1:
        K_expanded = K.repeat_interleave(num_key_value_groups, dim=1)
        V_expanded = V.repeat_interleave(num_key_value_groups, dim=1)
        return F.scaled_dot_product_attention(Q, K_expanded, V_expanded, is_causal=False)
    else:
        return F.scaled_dot_product_attention(Q, K, V, is_causal=False)

def flush_l2_cache(device):
    """通过写入 80MB 的垃圾数据，强制冲刷 A100 (40MB L2) 的缓存残留"""
    dummy = torch.empty(20 * 1024 * 1024, dtype=torch.float32, device=device)
    dummy.zero_()

class ModelProfiler:
    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype):
        self.model_name = model_name
        self.params = cfg.SUPPORTED_MODELS[model_name]
        self.device = device
        self.dtype = dtype
        self.paths = cfg.get_lut_paths(model_name)
        
        self.q_heads = self.params["q_heads"]
        self.kv_heads = self.params["kv_heads"]
        self.head_dim = self.params["head_dim"]
        self.d_model = self.params["d_model"]

        max_b = max(cfg.BUCKETS)
        self.pool_Q = torch.randn(cfg.BATCH_SIZE, self.q_heads, max_b, self.head_dim, device=self.device, dtype=self.dtype)
        self.pool_K = torch.randn(cfg.BATCH_SIZE, self.kv_heads, max_b, self.head_dim, device=self.device, dtype=self.dtype)
        self.pool_V = torch.randn(cfg.BATCH_SIZE, self.kv_heads, max_b, self.head_dim, device=self.device, dtype=self.dtype)
        self.pool_X = torch.randn(cfg.BATCH_SIZE, max_b, self.d_model, device=self.device, dtype=self.dtype)
        self.pool_W_A = torch.randn(cfg.BATCH_SIZE, self.d_model, cfg.LORA_RANK, device=self.device, dtype=self.dtype)
        self.pool_W_B = torch.randn(cfg.BATCH_SIZE, cfg.LORA_RANK, self.d_model, device=self.device, dtype=self.dtype)
        self.pool_Out = torch.empty(cfg.BATCH_SIZE, max_b, self.d_model, device=self.device, dtype=self.dtype)

    def measure_solo_time(self, S: int) -> float:
        Q, K, V = self.pool_Q[:, :, :S, :], self.pool_K[:, :, :S, :], self.pool_V[:, :, :S, :]
        X_lora, Out_lora = self.pool_X[:, :S, :], self.pool_Out[:, :S, :]
        W_A, W_B = self.pool_W_A, self.pool_W_B

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            for _ in range(WARMUP_ITERS): 
                simulate_attention(Q, K, V, self.q_heads, self.kv_heads)
                torch.bmm(torch.bmm(X_lora, W_A), W_B, out=Out_lora)
            torch.cuda.synchronize()
            
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            total_time_ms = 0.0
            
            for _ in range(ACTIVE_ITERS):
                # 1. 物理隔离：清空缓存并等待清空动作绝对完成
                flush_l2_cache(self.device)
                torch.cuda.synchronize() 
                
                # 2. 纯粹算子测速
                start.record()
                simulate_attention(Q, K, V, self.q_heads, self.kv_heads)
                torch.bmm(torch.bmm(X_lora, W_A), W_B, out=Out_lora)
                end.record()
                
                # 3. 结果累加
                torch.cuda.synchronize()
                total_time_ms += start.elapsed_time(end)
            
        return (total_time_ms * 1000) / ACTIVE_ITERS

    def measure_concurrent_time(self, S_s: int, S_c: int) -> float:
        Q_s, K_s, V_s = self.pool_Q[:, :, :S_s, :], self.pool_K[:, :, :S_s, :], self.pool_V[:, :, :S_s, :]
        X_s, Out_s = self.pool_X[:, :S_s, :], self.pool_Out[:, :S_s, :]
        Q_c, K_c, V_c = self.pool_Q[:, :, :S_c, :], self.pool_K[:, :, :S_c, :], self.pool_V[:, :, :S_c, :]
        X_c, Out_c = self.pool_X[:, :S_c, :], self.pool_Out[:, :S_c, :]
        W_A, W_B = self.pool_W_A, self.pool_W_B

        s_main = torch.cuda.current_stream()
        s_short = torch.cuda.Stream(device=self.device, priority=-1)
        s_long = torch.cuda.Stream(device=self.device, priority=0)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            for _ in range(WARMUP_ITERS):
                s_short.wait_stream(s_main)
                s_long.wait_stream(s_main)
                with torch.cuda.stream(s_short):
                    simulate_attention(Q_s, K_s, V_s, self.q_heads, self.kv_heads)
                    torch.bmm(torch.bmm(X_s, W_A), W_B, out=Out_s)
                with torch.cuda.stream(s_long):
                    simulate_attention(Q_c, K_c, V_c, self.q_heads, self.kv_heads)
                    torch.bmm(torch.bmm(X_c, W_A), W_B, out=Out_c)
                s_main.wait_stream(s_short)
                s_main.wait_stream(s_long)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end_short = torch.cuda.Event(enable_timing=True)
            total_time_ms = 0.0
            
            for _ in range(ACTIVE_ITERS):
                # 物理隔离
                flush_l2_cache(self.device)
                torch.cuda.synchronize()
                
                start.record(s_main) # 记录起跑线
                
                s_short.wait_stream(s_main)
                s_long.wait_stream(s_main)
                
                with torch.cuda.stream(s_short):
                    simulate_attention(Q_s, K_s, V_s, self.q_heads, self.kv_heads)
                    torch.bmm(torch.bmm(X_s, W_A), W_B, out=Out_s)
                    end_short.record(s_short) # 记录短任务独立冲线时间
                        
                with torch.cuda.stream(s_long):
                    simulate_attention(Q_c, K_c, V_c, self.q_heads, self.kv_heads)
                    torch.bmm(torch.bmm(X_c, W_A), W_B, out=Out_c)
                    
                s_main.wait_stream(s_short)
                s_main.wait_stream(s_long)
                
                # 等待这一轮所有计算收敛，再累加短任务的时间
                torch.cuda.synchronize()
                total_time_ms += start.elapsed_time(end_short)
            
        return (total_time_ms * 1000) / ACTIVE_ITERS

    def measure_read_amp_sum(self, S_l: int, S_c: int) -> float:
        if S_c >= S_l: return 0.0
        k = S_l // S_c + (1 if S_l % S_c != 0 else 0)
        
        K_full = self.pool_K[:, :, :S_l, :]
        V_full = self.pool_V[:, :, :S_l, :]
        
        q_chunks = []
        history_lens = []
        for i in range(1, k):
            curr_len = min(S_c, S_l - i * S_c)
            history_lens.append(i * S_c + curr_len)
            q_chunks.append(self.pool_Q[:, :, :curr_len, :])

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            for _ in range(WARMUP_ITERS):
                for i in range(k - 1):
                    simulate_attention(q_chunks[i], K_full[:, :, :history_lens[i], :], V_full[:, :, :history_lens[i], :], self.q_heads, self.kv_heads)
            torch.cuda.synchronize()

            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            total_time_ms = 0.0
            
            for _ in range(ACTIVE_ITERS):
                # 物理隔离
                flush_l2_cache(self.device)
                torch.cuda.synchronize()
                
                # 记录这 k-1 个 Chunk 连续执行的总时间
                start.record()
                for i in range(k - 1):
                    simulate_attention(q_chunks[i], K_full[:, :, :history_lens[i], :], V_full[:, :, :history_lens[i], :], self.q_heads, self.kv_heads)
                end.record()
                
                torch.cuda.synchronize()
                total_time_ms += start.elapsed_time(end)
            
        return (total_time_ms * 1000) / ACTIVE_ITERS

    def run(self):
        print(f"\n{'='*70}")
        print(f"=== Profiling {self.model_name} ({self.params['attn_type']}) on {torch.cuda.get_device_name(self.device)} ===")
        print(f"Arch: Q_Heads={self.q_heads} | KV_Heads={self.kv_heads} | D_Model={self.d_model}")
        print(f"{'='*70}")
        
        profile_data = {"T_solo": {}, "T_conc": {}, "T_read_amp": {}}

        print("[1/3] 基线执行时间测定 (T_solo)...")
        for b in tqdm(cfg.BUCKETS, desc="T_solo"):
            profile_data["T_solo"][b] = self.measure_solo_time(b)

        print("[2/3] 短任务物理争用逃逸时间测定 (T_conc_s)...")
        for s_s in tqdm(cfg.BUCKETS, desc="T_conc"):
            profile_data["T_conc"][s_s] = {}
            for s_c in cfg.BUCKETS:
                if s_c >= s_s:
                    profile_data["T_conc"][s_s][s_c] = self.measure_concurrent_time(s_s, s_c)

        print("[3/3] L2 Cache 溢出边界与读放大耗时测定 (T_read_amp)...")
        for s_l in tqdm(cfg.BUCKETS, desc="T_read_amp"):
            profile_data["T_read_amp"][s_l] = {}
            for s_c in cfg.BUCKETS:
                profile_data["T_read_amp"][s_l][s_c] = self.measure_read_amp_sum(s_l, s_c) if s_c < s_l else 0.0

        with open(self.paths["raw"], "w") as f:
            json.dump(profile_data, f, indent=4)
        print(f"✅ [{self.model_name}] 原始物理数据已保存至: {self.paths['raw']}")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave-Slice offline profiler")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys in config/hw_config.py, or 'all'.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
    )
    parser.add_argument("--warmup-iters", type=int, default=WARMUP_ITERS)
    parser.add_argument("--active-iters", type=int, default=ACTIVE_ITERS)
    parser.add_argument(
        "--buckets",
        default=None,
        help="Optional comma-separated buckets override, e.g. 32,64,128,256,512,1024,2048,4096",
    )
    parser.add_argument(
        "--serialize-gpu-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Serialize GPU-backed profiling runs through a global file lock.",
    )
    parser.add_argument(
        "--gpu-lock-path",
        default="",
        help="Optional file path used for the global GPU experiment lock.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not torch.cuda.is_available():
        print("Fatal Error: 本脚本强依赖 NVIDIA GPU 环境，当前检测为 CPU。")
        sys.exit(1)

    if args.buckets:
        cfg.BUCKETS = [int(x.strip()) for x in args.buckets.split(",") if x.strip()]
        cfg.BUCKETS = sorted({b for b in cfg.BUCKETS if b > 0})
        if not cfg.BUCKETS:
            raise ValueError("Invalid --buckets")

    WARMUP_ITERS = max(1, int(args.warmup_iters))
    ACTIVE_ITERS = max(1, int(args.active_iters))

    model_names = list(cfg.SUPPORTED_MODELS.keys())
    if args.models.strip().lower() != "all":
        chosen = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in chosen if m not in cfg.SUPPORTED_MODELS]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}")
        model_names = chosen

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.cuda.empty_cache()

    with gpu_experiment_lock(
        label=f"offline_profiler:{','.join(model_names)}",
        enabled=bool(args.serialize_gpu_tests),
        lock_path=args.gpu_lock_path or None,
    ):
        print(f"[Profiler] models={model_names}")
        print(f"[Profiler] buckets={cfg.BUCKETS}")
        print(f"[Profiler] warmup={WARMUP_ITERS} active={ACTIVE_ITERS}")
        for model_name in model_names:
            profiler = ModelProfiler(model_name, device, dtype)
            profiler.run()

        print("\n🎉 选定模型的物理底座采集完毕！")
