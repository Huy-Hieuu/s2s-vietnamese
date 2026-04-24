"""Tests for the ASR module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.asr.config import ASRConfig, StreamingConfig, load_asr_config
from src.asr.model import WhisperWrapper
from src.asr.streaming import PartialTranscript, StreamingASR


class TestASRConfig:
    def test_default_config(self) -> None:
        config = ASRConfig()
        assert config.model.language == "vi"
        assert config.model.use_flash_attention_2 is True
        assert config.data.sampling_rate == 16000

    def test_load_config(self, tmp_path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "model:\n  name: test-model\n  language: vi\n"
            "streaming:\n  chunk_size_s: 2.0\n"
        )
        config = load_asr_config(config_file)
        assert config.model.name == "test-model"
        assert config.streaming.chunk_size_s == 2.0


class TestWhisperWrapper:
    def test_not_loaded_by_default(self) -> None:
        config = ASRConfig()
        model = WhisperWrapper(config)
        assert not model.is_loaded

    def test_transcribe_raises_when_not_loaded(self, sample_audio_float32: NDArray) -> None:
        config = ASRConfig()
        model = WhisperWrapper(config)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.transcribe(sample_audio_float32)


class TestStreamingASR:
    @pytest.mark.asyncio
    async def test_transcribe_stream_yields_final(
        self, mock_asr_model, sample_audio_5s: NDArray
    ) -> None:
        config = ASRConfig(streaming=StreamingConfig(chunk_size_s=3.0, stride_s=1.0))
        streamer = StreamingASR(config, mock_asr_model)

        chunks = [sample_audio_5s[:16000], sample_audio_5s[16000:]]

        async def audio_gen():
            for c in chunks:
                yield c

        results = []
        async for transcript in streamer.transcribe_stream(audio_gen()):
            results.append(transcript)

        assert len(results) >= 1
        assert results[-1].is_final is True
        assert all(isinstance(r, PartialTranscript) for r in results)

    @pytest.mark.asyncio
    async def test_reset_clears_buffer(self, mock_asr_model) -> None:
        config = ASRConfig()
        streamer = StreamingASR(config, mock_asr_model)
        streamer._buffer = np.ones(1000, dtype=np.float32)
        streamer.reset()
        assert len(streamer._buffer) == 0
