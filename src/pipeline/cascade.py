"""Cascaded pipeline: ASR → LLM → TTS wiring with streaming.

This is the integration layer — it contains no ML logic itself.
It composes the three core modules into a streaming chain.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.asr.model import WhisperWrapper
from src.asr.streaming import PartialTranscript, StreamingASR
from src.llm.serve import LLMServer
from src.tts.model import CosyVoiceWrapper
from src.tts.streaming import StreamingTTS
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineEvent:
    """An event emitted by the cascade pipeline."""

    type: str  # "transcript", "token", "audio", "error"
    data: Any = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class PipelineLatency:
    """Accumulated latency metrics for a pipeline run."""

    vad_ms: float = 0.0
    asr_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    e2e_ms: float = 0.0


class CascadePipeline:
    """Wires ASR → LLM → TTS into a streaming cascade pipeline.

    Takes raw audio from VAD, runs it through ASR for transcription,
    feeds the transcript to the LLM, and synthesizes the LLM response
    with TTS — all streaming, all async.
    """

    def __init__(
        self,
        asr_model: WhisperWrapper,
        llm_server: LLMServer,
        tts_model: CosyVoiceWrapper,
        *,
        asr_config: Any = None,
        tts_config: Any = None,
        system_prompt: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        default_speaker: str = "default",
    ) -> None:
        self._asr = StreamingASR(asr_config, asr_model) if asr_config else None
        self._llm = llm_server
        self._tts = StreamingTTS(tts_config, tts_model) if tts_config else None
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.default_speaker = default_speaker

    async def process(
        self,
        audio_iter: AsyncIterator[NDArray[np.float32]],
    ) -> AsyncIterator[PipelineEvent]:
        """Run the full cascade pipeline on an audio stream.

        Args:
            audio_iter: Async iterator of float32 audio chunks (post-VAD).

        Yields:
            PipelineEvent at each stage: partial transcripts, tokens, audio chunks.
        """
        pipeline_start = time.perf_counter()
        latency = PipelineLatency()

        # Stage 1: ASR — audio → text
        async for transcript in self._run_asr(audio_iter):
            latency.asr_ms = max(latency.asr_ms, transcript.latency_ms)
            yield PipelineEvent(
                type="transcript",
                data=transcript.text,
                latency_ms=transcript.latency_ms,
            )

            if transcript.is_final:
                full_text = transcript.text
                break
        else:
            return  # No speech detected

        # Stage 2: LLM — text → response tokens
        prompt = self._format_prompt(full_text)

        async def _token_generator() -> AsyncIterator[str]:
            first_token = True
            async for chunk in self._llm.generate_stream(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            ):
                if first_token:
                    latency.llm_first_token_ms = chunk.latency_ms
                    first_token = False
                yield chunk.text
                yield PipelineEvent(type="token", data=chunk.text, latency_ms=chunk.latency_ms)

        # Stage 3: TTS — tokens → audio
        # Feed LLM tokens into TTS while streaming both
        token_queue: list[str] = []

        async def _text_source() -> AsyncIterator[str]:
            """Buffer tokens and yield them to TTS."""
            first_token = True
            async for chunk in self._llm.generate_stream(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            ):
                if first_token:
                    latency.llm_first_token_ms = chunk.latency_ms
                    first_token = False
                if chunk.text:
                    token_queue.append(chunk.text)
                    yield chunk.text

        if self._tts is not None:
            async for audio_chunk in self._tts.synthesize_stream(
                _text_source(), speaker_id=self.default_speaker,
            ):
                if latency.tts_first_chunk_ms == 0:
                    latency.tts_first_chunk_ms = audio_chunk.latency_ms
                latency.e2e_ms = (time.perf_counter() - pipeline_start) * 1000
                yield PipelineEvent(
                    type="audio",
                    data=audio_chunk.audio,
                    latency_ms=latency.e2e_ms,
                )

        latency.e2e_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            "pipeline_complete",
            asr_ms=f"{latency.asr_ms:.1f}",
            llm_first_token_ms=f"{latency.llm_first_token_ms:.1f}",
            tts_first_chunk_ms=f"{latency.tts_first_chunk_ms:.1f}",
            e2e_ms=f"{latency.e2e_ms:.1f}",
        )

    async def _run_asr(
        self,
        audio_iter: AsyncIterator[NDArray[np.float32]],
    ) -> AsyncIterator[PartialTranscript]:
        """Run ASR on audio stream. Yields partial and final transcripts."""
        if self._asr is None:
            yield PartialTranscript(text="", is_final=True)
            return
        async for transcript in self._asr.transcribe_stream(audio_iter):
            yield transcript

    def _format_prompt(self, user_text: str) -> str:
        """Format the user's transcription into an LLM prompt."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        # Use the LLM data module's formatter if available
        from src.llm.data import format_chat_llama3

        return format_chat_llama3(messages)
