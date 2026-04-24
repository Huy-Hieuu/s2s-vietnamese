"""Streaming ASR — async generator yielding partial transcripts."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.asr.config import ASRConfig
from src.asr.model import WhisperWrapper
from src.utils.audio import resample
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PartialTranscript:
    """A partial or final transcript from the ASR stream."""

    text: str
    is_final: bool = False
    confidence: float = 0.0
    latency_ms: float = 0.0
    language: str = "vi"


class StreamingASR:
    """Streaming ASR that processes audio chunks and yields partial transcripts.

    Uses a sliding-window approach: audio is buffered, and the model runs
    inference on overlapping chunks for improved accuracy at chunk boundaries.
    """

    def __init__(self, config: ASRConfig, model: WhisperWrapper) -> None:
        self.config = config
        self.model = model
        self._buffer: NDArray[np.float32] = np.array([], dtype=np.float32)
        self._chunk_samples = int(config.streaming.chunk_size_s * config.data.sampling_rate)
        self._stride_samples = int(config.streaming.stride_s * config.data.sampling_rate)

    async def transcribe_stream(
        self,
        audio_iter: AsyncIterator[NDArray[np.float32]],
    ) -> AsyncIterator[PartialTranscript]:
        """Process an audio stream and yield partial transcripts.

        Args:
            audio_iter: Async iterator of float32 audio chunks.

        Yields:
            PartialTranscript with incremental recognition results.
        """
        async for chunk in audio_iter:
            chunk = resample(chunk, 16000, self.config.data.sampling_rate)
            self._buffer = np.concatenate([self._buffer, chunk])

            while len(self._buffer) >= self._chunk_samples:
                start = time.perf_counter()
                audio_segment = self._buffer[: self._chunk_samples]
                text = self.model.transcribe(audio_segment, self.config.data.sampling_rate)
                latency = (time.perf_counter() - start) * 1000

                yield PartialTranscript(
                    text=text,
                    is_final=False,
                    latency_ms=latency,
                )
                self._buffer = self._buffer[self._stride_samples :]

        # Flush remaining buffer
        if len(self._buffer) > 0:
            start = time.perf_counter()
            text = self.model.transcribe(self._buffer, self.config.data.sampling_rate)
            latency = (time.perf_counter() - start) * 1000
            yield PartialTranscript(text=text, is_final=True, latency_ms=latency)

    def reset(self) -> None:
        """Clear the internal audio buffer."""
        self._buffer = np.array([], dtype=np.float32)
