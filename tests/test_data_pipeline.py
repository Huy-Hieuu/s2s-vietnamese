"""Tests for data pipeline utilities and audio processing."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.audio import (
    async_chunk_reader,
    chunk_audio,
    resample,
    to_float32,
    to_int16,
    to_wav_bytes,
)


class TestAudioConversions:
    def test_to_float32_from_int16(self) -> None:
        int_audio = np.array([0, 16384, 32767, -16384, -32768], dtype=np.int16)
        float_audio = to_float32(int_audio)
        assert float_audio.dtype == np.float32
        assert np.isclose(float_audio[2], 32767 / 32768, atol=1e-4)

    def test_to_float32_passthrough(self) -> None:
        float_in = np.array([0.5, -0.5], dtype=np.float32)
        float_out = to_float32(float_in)
        assert float_out.dtype == np.float32
        np.testing.assert_array_equal(float_in, float_out)

    def test_to_int16_roundtrip(self) -> None:
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        int_audio = to_int16(original)
        assert int_audio.dtype == np.int16
        assert int_audio[0] == 0
        assert int_audio[3] == 32767
        assert int_audio[4] == -32767


class TestResample:
    def test_same_rate_passthrough(self) -> None:
        audio = np.random.randn(16000).astype(np.float32)
        result = resample(audio, 16000, 16000)
        np.testing.assert_array_equal(result, audio)

    def test_downsample_halves_length(self) -> None:
        audio = np.random.randn(16000).astype(np.float32)
        result = resample(audio, 16000, 8000)
        assert abs(len(result) - 8000) <= 2


class TestWavBytes:
    def test_produces_valid_wav(self, sample_audio_float32: NDArray) -> None:
        wav = to_wav_bytes(sample_audio_float32, 16000)
        assert isinstance(wav, bytes)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"


class TestChunkAudio:
    def test_non_overlapping_chunks(self) -> None:
        audio = np.arange(100, dtype=np.float32)
        chunks = chunk_audio(audio, chunk_size=30)
        assert len(chunks) == 4
        assert len(chunks[0]) == 30
        assert len(chunks[-1]) == 10  # remainder

    def test_overlapping_chunks(self) -> None:
        audio = np.arange(100, dtype=np.float32)
        chunks = chunk_audio(audio, chunk_size=30, hop=10)
        assert len(chunks) == 10


class TestAsyncChunkReader:
    @pytest.mark.asyncio
    async def test_yields_chunks(self) -> None:
        audio = np.arange(100, dtype=np.float32)
        chunks = []
        async for chunk in async_chunk_reader(audio, chunk_size=30):
            chunks.append(chunk)
        assert len(chunks) == 4
