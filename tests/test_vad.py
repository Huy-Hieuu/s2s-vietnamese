"""Tests for the VAD module."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.pipeline.vad import VoiceActivityDetector


class TestVoiceActivityDetector:
    def test_init_defaults(self) -> None:
        vad = VoiceActivityDetector()
        assert vad.threshold == 0.5
        assert vad.sampling_rate == 16000

    def test_speech_prob_fallback(self, sample_audio_float32: NDArray) -> None:
        vad = VoiceActivityDetector()
        prob = vad._get_speech_prob(sample_audio_float32)
        assert 0.0 <= prob <= 1.0

    @pytest.mark.asyncio
    async def test_silence_not_yielded(self) -> None:
        vad = VoiceActivityDetector(threshold=0.99)

        silence = np.zeros(16000, dtype=np.float32)

        async def audio_gen():
            yield silence

        segments = []
        async for seg in vad.detect_speech(audio_gen()):
            segments.append(seg)

        assert len(segments) == 0

    @pytest.mark.asyncio
    async def test_loud_audio_detected(self, sample_audio_5s: NDArray) -> None:
        vad = VoiceActivityDetector(threshold=0.01, min_speech_duration_ms=100)

        async def audio_gen():
            yield sample_audio_5s

        segments = []
        async for seg in vad.detect_speech(audio_gen()):
            segments.append(seg)

        assert len(segments) >= 1
        for seg in segments:
            assert seg.duration_ms > 0
