"""Tests for the TTS module."""

from __future__ import annotations

import pytest

from src.tts.config import StreamingConfig as TTSStreamingConfig
from src.tts.config import TTSConfig, load_tts_config
from src.tts.model import CosyVoiceWrapper
from src.tts.streaming import AudioChunk, StreamingTTS


class TestTTSConfig:
    def test_default_config(self) -> None:
        config = TTSConfig()
        assert config.model.sample_rate == 24000
        assert config.voice_cloning.enabled is True

    def test_load_config(self, tmp_path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "model:\n  sample_rate: 22050\ntraining:\n  num_train_epochs: 5\n"
        )
        config = load_tts_config(config_file)
        assert config.model.sample_rate == 22050
        assert config.training.num_train_epochs == 5


class TestCosyVoiceWrapper:
    def test_not_loaded_by_default(self) -> None:
        config = TTSConfig()
        model = CosyVoiceWrapper(config)
        assert not model.is_loaded

    def test_synthesize_raises_when_not_loaded(self) -> None:
        config = TTSConfig()
        model = CosyVoiceWrapper(config)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.synthesize("Xin chào")


class TestStreamingTTS:
    @pytest.mark.asyncio
    async def test_synthesize_stream(self, mock_tts_model) -> None:
        config = TTSConfig(streaming=TTSStreamingConfig(chunk_size_tokens=10))
        tts = StreamingTTS(config, mock_tts_model)

        async def text_gen():
            for word in ["Xin", " chào", " bạn"]:
                yield word

        results = []
        async for chunk in tts.synthesize_stream(text_gen(), speaker_id="default"):
            results.append(chunk)

        assert len(results) >= 1
        assert all(isinstance(r, AudioChunk) for r in results)
        # Buffer is flushed at end of stream → last chunk is final
        assert results[-1].is_final is True

    @pytest.mark.asyncio
    async def test_reset_clears_buffer(self, mock_tts_model) -> None:
        config = TTSConfig()
        tts = StreamingTTS(config, mock_tts_model)
        tts._text_buffer = "Xin chào"
        tts.reset()
        assert tts._text_buffer == ""
