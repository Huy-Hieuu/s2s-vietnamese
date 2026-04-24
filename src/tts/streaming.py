"""Streaming TTS — async generator yielding audio chunks."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tts.config import TTSConfig
from src.tts.model import CosyVoiceWrapper
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioChunk:
    """An audio chunk from the TTS stream."""

    audio: NDArray[np.float32]
    sample_rate: int
    is_final: bool = False
    latency_ms: float = 0.0


class StreamingTTS:
    """Streaming TTS that converts text tokens to audio chunks.

    Synthesizes audio chunk-by-chunk as text streams in, enabling
    low-latency end-to-end pipeline output.
    """

    def __init__(self, config: TTSConfig, model: CosyVoiceWrapper) -> None:
        self.config = config
        self.model = model
        self._text_buffer: str = ""
        self._sentence_end_chars = {".", "!", "?", "。", "！", "？"}

    async def synthesize_stream(
        self,
        text_iter: AsyncIterator[str],
        speaker_id: str = "default",
    ) -> AsyncIterator[AudioChunk]:
        """Convert a stream of text tokens into a stream of audio chunks.

        Buffers incoming text until a sentence boundary is detected, then
        synthesizes and yields the resulting audio.

        Args:
            text_iter: Async iterator of text token strings from the LLM.
            speaker_id: Speaker identity for multi-speaker TTS.

        Yields:
            AudioChunk with synthesized audio data.
        """
        async for token in text_iter:
            self._text_buffer += token

            if self._should_synthesize():
                async for chunk in self._synthesize_buffer(speaker_id):
                    yield chunk

        # Flush remaining text
        if self._text_buffer.strip():
            async for chunk in self._synthesize_buffer(speaker_id, final=True):
                yield chunk

    def _should_synthesize(self) -> bool:
        """Check if the buffer contains enough text to synthesize a chunk."""
        if not self._text_buffer:
            return False
        # Synthesize on sentence boundaries
        if self._text_buffer[-1] in self._sentence_end_chars:
            return True
        # Or when buffer is large enough for a phrase
        chunk_tokens = self.config.streaming.chunk_size_tokens
        if len(self._text_buffer.split()) >= chunk_tokens:
            return True
        return False

    async def _synthesize_buffer(
        self,
        speaker_id: str,
        final: bool = False,
    ) -> AsyncIterator[AudioChunk]:
        """Synthesize the current text buffer and yield audio."""
        text = self._text_buffer.strip()
        self._text_buffer = ""

        if not text:
            return

        start = time.perf_counter()
        audio = self.model.synthesize(text, speaker_id=speaker_id)
        latency = (time.perf_counter() - start) * 1000

        yield AudioChunk(
            audio=audio,
            sample_rate=self.config.model.sample_rate,
            is_final=final,
            latency_ms=latency,
        )

    def reset(self) -> None:
        """Clear the text buffer."""
        self._text_buffer = ""
