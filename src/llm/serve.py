"""vLLM-based LLM serving with async streaming."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from src.llm.config import LLMConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenChunk:
    """A chunk of generated tokens from the LLM."""

    text: str
    token_ids: list[int]
    is_final: bool = False
    latency_ms: float = 0.0


class LLMServer:
    """Wraps vLLM's AsyncLLMEngine for production streaming inference.

    Do not bypass vLLM to call the model directly — continuous batching
    is critical for throughput under concurrent load.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._engine: object | None = None

    async def start(self) -> None:
        """Initialize the vLLM async engine."""
        serving = self.config.serving
        logger.info(
            "starting_llm_engine",
            model=self.config.model.name,
            tp=serving.tensor_parallel_size,
            gpu_util=serving.gpu_memory_utilization,
        )
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine

            engine_args = AsyncEngineArgs(
                model=self.config.model.name,
                tensor_parallel_size=serving.tensor_parallel_size,
                gpu_memory_utilization=serving.gpu_memory_utilization,
                max_model_len=self.config.model.max_model_len,
                max_num_seqs=serving.max_num_seqs,
                max_num_batched_tokens=serving.max_num_batched_tokens,
                enable_prefix_caching=serving.enable_prefix_caching,
                dtype=self.config.model.torch_dtype,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("llm_engine_started")
        except ImportError:
            logger.warning("vllm_not_installed_fallback")
            self._engine = None

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncIterator[TokenChunk]:
        """Stream generated tokens from the LLM.

        Args:
            prompt: Input prompt string.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling.

        Yields:
            TokenChunk with incremental generated text.
        """
        start_time = time.perf_counter()

        if self._engine is not None:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            request_id = f"req-{time.monotonic_ns()}"
            results_generator = self._engine.generate(prompt, sampling_params, request_id)

            full_text = ""
            async for request_output in results_generator:
                for output in request_output.outputs:
                    new_text = output.text[len(full_text) :]
                    if new_text:
                        full_text += new_text
                        latency = (time.perf_counter() - start_time) * 1000
                        yield TokenChunk(
                            text=new_text,
                            token_ids=[],
                            latency_ms=latency,
                        )
            yield TokenChunk(
                text="",
                token_ids=[],
                is_final=True,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        else:
            # Fallback for when vLLM is not installed
            yield TokenChunk(
                text="[LLM response placeholder]",
                token_ids=[],
                is_final=True,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

    async def stop(self) -> None:
        """Shutdown the engine."""
        if self._engine is not None and hasattr(self._engine, "shutdown"):
            await self._engine.shutdown()
        self._engine = None
        logger.info("llm_engine_stopped")
