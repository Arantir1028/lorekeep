"""Entry point used by vLLM to initialize WaveSlice in child processes."""


def register() -> None:
    from waveslice.vllm.integration import activate_from_environment

    activate_from_environment()


__all__ = ["register"]
