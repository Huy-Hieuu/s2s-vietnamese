"""Shared test fixtures for S2S tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def sample_audio_float32() -> NDArray[np.float32]:
    """1 second of synthetic float32 audio at 16kHz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


@pytest.fixture
def sample_audio_int16(sample_audio_float32: NDArray[np.float32]) -> NDArray[np.int16]:
    """1 second of synthetic int16 audio at 16kHz."""
    return (sample_audio_float32 * 32767).astype(np.int16)


@pytest.fixture
def sample_audio_5s() -> NDArray[np.float32]:
    """5 seconds of synthetic float32 audio at 16kHz."""
    sr = 16000
    t = np.linspace(0, 5.0, sr * 5, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


@pytest.fixture
def mock_asr_model():
    """Mock WhisperWrapper that returns a fixed Vietnamese transcript."""
    model = MagicMock()
    model.is_loaded = True
    model.transcribe.return_value = "Xin chào, tôi là trợ lý AI."
    return model


@pytest.fixture
def mock_llm_server():
    """Mock LLMServer that streams predefined tokens."""
    server = MagicMock()

    async def _fake_stream(*args, **kwargs):
        tokens = ["Xin", " chào", " bạn", "."]
        for i, tok in enumerate(tokens):
            yield MagicMock(
                text=tok,
                token_ids=[],
                is_final=(i == len(tokens) - 1),
                latency_ms=50.0 * (i + 1),
            )

    server.generate_stream = _fake_stream
    server.start = AsyncMock()
    server.stop = AsyncMock()
    return server


@pytest.fixture
def mock_tts_model():
    """Mock CosyVoiceWrapper that returns synthetic audio."""
    model = MagicMock()
    model.is_loaded = True
    model.synthesize.return_value = np.random.randn(24000).astype(np.float32) * 0.1
    return model
