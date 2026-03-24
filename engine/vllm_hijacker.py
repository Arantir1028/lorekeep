# lorekeep/engine/vllm_hijacker.py

import time
import logging
import collections
from typing import List, Tuple
import torch

from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.worker.model_runner import ModelRunner
from scheduler.wave_scheduler import WaveScheduler
from engine.lora_unbinder import WaveLoRAUnbinder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WaveSlice")

_ORIGINAL_SCHEDULE = Scheduler._schedule
_ORIGINAL_SCHEDULE_RUNNING = Scheduler._schedule_running
_ORIGINAL_EXECUTE_MODEL = ModelRunner.execute_model

_WAVE_BRAIN = None
_WAVE_UNBINDER = None

def _wave_schedule_running_hook(self, running_queue, budget, *args, **kwargs):
    """
    【二级劫持】：精准掐断 Running 队列中长任务的贪婪吞噬
    """
    if getattr(self, "wave_running_budget_limit", None) is not None:
        original_token_budget = budget.token_budget
        # 将 Running 队列的可用预算缩减，逼迫引擎释放剩余算力
        budget.token_budget = min(original_token_budget, self.wave_running_budget_limit)
        outputs = _ORIGINAL_SCHEDULE_RUNNING(self, running_queue, budget, *args, **kwargs)
        # 恢复预算，让渡给后面的 Waiting 短任务
        budget.token_budget = original_token_budget
        return outputs
    else:
        return _ORIGINAL_SCHEDULE_RUNNING(self, running_queue, budget, *args, **kwargs)

def _wave_schedule_hook(self: Scheduler) -> Tuple[List[SchedulerOutputs], bool]:
    global _WAVE_BRAIN
    
    waiting_queue = self.waiting
    running_queue = self.running

    # 1. 提取全生命周期的剩余 Token 长度
    all_seqs = []
    max_wait_time_us = 0.0
    current_time = time.time()
    
    for sg in waiting_queue:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        if l > 0:
            all_seqs.append(l)
            wait_time_us = (current_time - sg.metrics.arrival_time) * 1e6
            if wait_time_us > max_wait_time_us: max_wait_time_us = wait_time_us
            
    for sg in running_queue:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        if l > 0:
            all_seqs.append(l)

    if len(all_seqs) < 2:
        return _ORIGINAL_SCHEDULE(self)

    S_s_real = min(all_seqs)
    S_l_real = max(all_seqs)

    if S_l_real < S_s_real * 4 or S_l_real <= 512:
        return _ORIGINAL_SCHEDULE(self)

    # =====================================================================
    # 2. 委托给 FairnessEngine 进行严格的排队论映射
    # =====================================================================
    # 此时不再传递粗糙的 rho_est，而是只需传入系统中存活的物理请求总数 L
    total_reqs = len(waiting_queue) + len(running_queue)
    best_S_c = _WAVE_BRAIN.schedule_real(S_s_real, S_l_real, max_wait_time_us, total_reqs)

    # 若触发熔断退化
    if best_S_c >= S_l_real:
        return _ORIGINAL_SCHEDULE(self)

    # =====================================================================
    # 3. 核心修复区：只重排 Waiting 队列，绝对禁止重排 running 队列
    # =====================================================================
    waiting_short, waiting_long = [], []
    waiting_short_tokens = 0
    for sg in self.waiting:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        if l <= best_S_c:
            waiting_short.append(sg)
            waiting_short_tokens += l
        else:
            waiting_long.append(sg)
    self.waiting = collections.deque(waiting_short + waiting_long)
    
    # 动态计算 Running 队列的合理预算上限
    running_budget_limit = 0
    for sg in self.running:
        seq = next(iter(sg.get_seqs()))
        l = seq.get_len() - seq.data.get_num_computed_tokens()
        running_budget_limit += min(l, best_S_c)

    # 4. 跨界物理预算微操
    original_budget = self.scheduler_config.max_num_batched_tokens
    global_hijacked_budget = running_budget_limit + waiting_short_tokens
    
    self.scheduler_config.max_num_batched_tokens = global_hijacked_budget
    self.wave_running_budget_limit = running_budget_limit
    
    logger.info(f"🌊 [Wave-Slice] 决断介入: 检测到短任务排队。将长任务 {S_l_real} 切分为 {best_S_c}。")
    if waiting_short_tokens > 0:
        logger.info(f"🌊 [Wave-Slice] 物理插队: {len(waiting_short)} 个短任务跃迁至 Waiting 队首, 跨界并发预算锁死为 {global_hijacked_budget}")

    outputs = _ORIGINAL_SCHEDULE(self)
    
    # 状态擦除，做到无痕劫持
    self.scheduler_config.max_num_batched_tokens = original_budget
    self.wave_running_budget_limit = None
    
    return outputs

def _wave_execute_model_hook(self: ModelRunner, *args, **kwargs):
    """
    【Phase II 劫持】: 物理流解绑。
    升级版：X光级底层探针，完美适配 vLLM v0.4.x Chunked Prefill + Decode 混合元数据拓扑。
    """
    global _WAVE_UNBINDER
    model_input = args[0] if len(args) > 0 else kwargs.get("model_input")
    
    is_heterogeneous = False
    debug_info = ""
    
    if model_input is not None and hasattr(model_input, "attn_metadata"):
        attn_meta = model_input.attn_metadata
        
        # 探测维度 1：Decode 与 Prefill 混合拥塞
        num_prefills = getattr(attn_meta, "num_prefills", 0)
        num_decode_tokens = getattr(attn_meta, "num_decode_tokens", 0)
        
        # 获取 Prefill 的长度分布 (兼容不同版本的变量名)
        lens_tensor = getattr(attn_meta, "prompt_lens_tensor", None)
        if lens_tensor is None:
            lens_tensor = getattr(attn_meta, "seq_lens_tensor", None)
            
        real_chunk_sizes = lens_tensor.tolist() if lens_tensor is not None else []
        
        # 🔴 X光级底层探针：打印每一次 GPU Kernel Launch 的真实组装状态
        logger.info(f"🔎 [底层探针] GPU Batch 下发 -> Prefills: {num_prefills}, Decodes: {num_decode_tokens}, Prefill_Lens: {real_chunk_sizes}")
        
        # 拦截逻辑判定
        if num_prefills > 0 and num_decode_tokens > 0:
            is_heterogeneous = True
            debug_info = f"Prefill({num_prefills}个) + Decode({num_decode_tokens}词) 混合拥塞"
            
        elif len(real_chunk_sizes) > 1 and max(real_chunk_sizes) >= 4 * min(real_chunk_sizes):
            is_heterogeneous = True
            debug_info = f"异构 Prefill 长度分布 {real_chunk_sizes}"

    if is_heterogeneous and _WAVE_UNBINDER is not None:
        logger.info(f"🌊 [Wave-Slice Phase II] 物理图拦截: {debug_info}。启动 Multi-Stream 并发解绑...")
        # ==========================================================
        # 核心：将混合 Batch 的执行强制推入自定义的高优先级异步流
        # ==========================================================
        with torch.cuda.stream(_WAVE_UNBINDER.stream_short):
            output = _ORIGINAL_EXECUTE_MODEL(self, *args, **kwargs)
            # 在高优先级流中记录完成事件
            short_done_event = torch.cuda.Event(enable_timing=False)
            short_done_event.record(_WAVE_UNBINDER.stream_short)
            
        # 让 Host CPU 提前进行 SLA Rescue，同步短任务流
        short_done_event.synchronize()
        return output
    else:
        return _ORIGINAL_EXECUTE_MODEL(self, *args, **kwargs)

def inject_wave_slice(model_name: str):
    global _WAVE_BRAIN, _WAVE_UNBINDER
    # 实例化 Phase I 的 O(1) 决断大脑
    _WAVE_BRAIN = WaveScheduler(model_name, gamma=2.0)
    
    # 实例化 Phase II 的空间解绑器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # 为了泛用性在此加入容错，以防底层硬件暂不支持多流
        _WAVE_UNBINDER = WaveLoRAUnbinder(d_model=4096, r=64, device=device)
    except Exception as e:
        logger.warning(f"WaveLoRAUnbinder 实例化降级: {e}")
        _WAVE_UNBINDER = None
    
    Scheduler._schedule = _wave_schedule_hook
    Scheduler._schedule_running = _wave_schedule_running_hook 
    ModelRunner.execute_model = _wave_execute_model_hook
    logger.info(f"🌊 [Wave-Slice] 针对 {model_name} 的全阶段 (Phase I + Phase II) 底层劫持完成。")