"""vLLM V1 scheduler adapter selected by :class:`waveslice.EngineArgs`."""

from waveslice.vllm.bootstrap import bootstrap_vllm_runtime

bootstrap_vllm_runtime()

from vllm.v1.core.sched.scheduler import Scheduler as VllmScheduler  # noqa: E402


class WaveSliceScheduler(VllmScheduler):
    """Isolate WaveSlice scheduler hooks from vLLM's default scheduler class."""


__all__ = ["WaveSliceScheduler"]
