"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SpeechSegment:
    """A speech segment detected by VAD."""

    audio: NDArray[np.float32]
    start_ms: float
    duration_ms: float
    confidence: float


class VoiceActivityDetector:
    """Silero VAD wrapper for gating audio chunks to the ASR module.

    Filters out silence and noise before forwarding speech segments
    downstream, reducing unnecessary ASR processing.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_ms: int = 30,
        sampling_rate: int = 16000,
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_ms = window_size_ms
        self.sampling_rate = sampling_rate
        self._model: object | None = None

    def load(self) -> None:
        """Load the Silero VAD model."""
        logger.info("loading_vad_model")
        try:
            import torch

            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            logger.info("vad_model_loaded")
        except Exception:
            logger.warning("vad_model_load_failed_fallback")
            self._model = None

    def _get_speech_prob(self, audio_chunk: NDArray[np.float32]) -> float:
        """Get speech probability for an audio window."""
        if self._model is not None:
            import torch

            with torch.no_grad():
                return float(self._model(torch.from_numpy(audio_chunk), self.sampling_rate).item())
        # Fallback: simple energy-based detection
        energy = np.sqrt(np.mean(audio_chunk**2))
        return min(energy * 20.0, 1.0)

    async def detect_speech(
        self,
        audio_stream: AsyncIterator[NDArray[np.float32]],
    ) -> AsyncIterator[SpeechSegment]:
        """Process an audio stream and yield speech segments.

        Buffers audio and yields segments when speech ends (after min_silence_duration
        of silence following min_speech_duration of speech).

        Args:
            audio_stream: Async iterator of float32 audio chunks.

        Yields:
            SpeechSegment with the detected speech audio.
        """
        window_samples = int(self.window_size_ms * self.sampling_rate / 1000)
        min_speech_samples = int(self.min_speech_duration_ms * self.sampling_rate / 1000)
        min_silence_samples = int(self.min_silence_duration_ms * self.sampling_rate / 1000)

        speech_buffer: list[NDArray[np.float32]] = []
        silence_count = 0
        is_speaking = False
        total_offset = 0

        async for chunk in audio_stream:
            # Process chunk in windows
            for start in range(0, len(chunk), window_samples):
                window = chunk[start : start + window_samples]
                if len(window) < window_samples // 2:
                    break

                prob = self._get_speech_prob(window)
                is_speech = prob >= self.threshold

                if is_speech:
                    speech_buffer.append(window)
                    silence_count = 0
                    is_speaking = True
                elif is_speaking:
                    silence_count += len(window)
                    if silence_count >= min_silence_samples:
                        # Speech segment ended
                        audio = np.concatenate(speech_buffer)
                        if len(audio) >= min_speech_samples:
                            duration_ms = len(audio) / self.sampling_rate * 1000
                            yield SpeechSegment(
                                audio=audio,
                                start_ms=total_offset / self.sampling_rate * 1000,
                                duration_ms=duration_ms,
                                confidence=prob,
                            )
                        speech_buffer.clear()
                        is_speaking = False
                        silence_count = 0

                total_offset += len(window)

        # Flush remaining speech
        if speech_buffer:
            audio = np.concatenate(speech_buffer)
            if len(audio) >= min_speech_samples:
                duration_ms = len(audio) / self.sampling_rate * 1000
                yield SpeechSegment(
                    audio=audio,
                    start_ms=total_offset / self.sampling_rate * 1000 - duration_ms,
                    duration_ms=duration_ms,
                    confidence=1.0,
                )
