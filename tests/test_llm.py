"""Tests for the LLM module."""

from __future__ import annotations

import pytest

from src.llm.config import LLMConfig, load_llm_config
from src.llm.data import format_chat_llama3
from src.llm.serve import LLMServer, TokenChunk


class TestLLMConfig:
    def test_default_config(self) -> None:
        config = LLMConfig()
        assert "llama" in config.model.name.lower()
        assert config.serving.engine == "vllm"

    def test_load_config(self, tmp_path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "model:\n  name: test-llm\nserving:\n  gpu_memory_utilization: 0.9\n"
        )
        config = load_llm_config(config_file)
        assert config.model.name == "test-llm"
        assert config.serving.gpu_memory_utilization == 0.9


class TestLLMServer:
    @pytest.mark.asyncio
    async def test_generate_stream_fallback(self) -> None:
        """Server yields a placeholder when vLLM is not installed."""
        config = LLMConfig()
        server = LLMServer(config)
        # Don't call start() — engine stays None

        results = []
        async for chunk in server.generate_stream("Xin chào"):
            results.append(chunk)

        assert len(results) >= 1
        assert isinstance(results[0], TokenChunk)

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        config = LLMConfig()
        server = LLMServer(config)
        await server.start()
        await server.stop()


class TestDataUtils:
    def test_format_chat_llama3(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = format_chat_llama3(messages)
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result

    def test_format_chat_no_generation_prompt(self) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        result = format_chat_llama3(messages, add_generation_prompt=False)
        assert "<|start_header_id|>assistant<|end_header_id|>" not in result
