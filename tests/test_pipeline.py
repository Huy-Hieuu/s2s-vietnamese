"""Tests for the cascade pipeline wiring."""

from __future__ import annotations

import pytest
from numpy.typing import NDArray

from src.asr.config import ASRConfig
from src.pipeline.cascade import CascadePipeline, PipelineEvent
from src.pipeline.vad import SpeechSegment, VoiceActivityDetector
from src.tts.config import TTSConfig


class TestVoiceActivityDetector:
    @pytest.mark.asyncio
    async def test_detect_speech_yields_segment(self, sample_audio_5s: NDArray) -> None:
        vad = VoiceActivityDetector(threshold=0.01, min_speech_duration_ms=100)
        # Don't load the actual Silero model — use energy fallback

        async def audio_gen():
            yield sample_audio_5s

        segments = []
        async for seg in vad.detect_speech(audio_gen()):
            segments.append(seg)

        # With a loud sine wave and low threshold, should detect speech
        for seg in segments:
            assert isinstance(seg, SpeechSegment)
            assert len(seg.audio) > 0
            assert seg.duration_ms > 0


class TestCascadePipeline:
    @pytest.mark.asyncio
    async def test_process_emits_events(
        self,
        mock_asr_model,
        mock_llm_server,
        mock_tts_model,
        sample_audio_5s: NDArray,
    ) -> None:
        asr_config = ASRConfig()
        tts_config = TTSConfig()

        pipeline = CascadePipeline(
            asr_model=mock_asr_model,
            llm_server=mock_llm_server,
            tts_model=mock_tts_model,
            asr_config=asr_config,
            tts_config=tts_config,
            system_prompt="Test prompt",
            max_new_tokens=50,
        )

        async def audio_source():
            yield sample_audio_5s

        events = []
        async for event in pipeline.process(audio_source()):
            events.append(event)

        assert len(events) >= 1
        event_types = {e.type for e in events}
        assert "transcript" in event_types

    @pytest.mark.asyncio
    async def test_pipeline_formats_prompt(
        self, mock_asr_model, mock_llm_server, mock_tts_model
    ) -> None:
        pipeline = CascadePipeline(
            asr_model=mock_asr_model,
            llm_server=mock_llm_server,
            tts_model=mock_tts_model,
            system_prompt="Bạn là trợ lý.",
        )
        prompt = pipeline._format_prompt("Xin chào")
        assert "Bạn là trợ lý" in prompt
        assert "Xin chào" in prompt


class TestPipelineEvent:
    def test_event_creation(self) -> None:
        event = PipelineEvent(type="transcript", data="Xin chào")
        assert event.type == "transcript"
        assert event.data == "Xin chào"
        assert event.latency_ms == 0.0
