"""WebSocket server for real-time S2S streaming."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.pipeline.cascade import CascadePipeline
from src.pipeline.vad import VoiceActivityDetector
from src.utils.audio import to_wav_bytes
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class RealtimeSession:
    """Manages a single WebSocket S2S session."""

    def __init__(
        self,
        websocket: WebSocket,
        pipeline: CascadePipeline,
        vad: VoiceActivityDetector,
        sampling_rate: int = 16000,
    ) -> None:
        self.websocket = websocket
        self.pipeline = pipeline
        self.vad = vad
        self.sampling_rate = sampling_rate

    async def run(self) -> None:
        """Main session loop: receive audio, run pipeline, send responses."""
        await self.websocket.accept()
        logger.info("ws_session_started")

        try:
            while True:
                data = await self.websocket.receive_bytes()
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                async def audio_source() -> Any:
                    """Single-chunk async generator for VAD input."""
                    yield audio

                # VAD gate
                has_speech = False
                async for segment in self.vad.detect_speech(audio_source()):
                    has_speech = True
                    async def speech_source() -> Any:
                        yield segment.audio

                    # Run cascade pipeline
                    async for event in self.pipeline.process(speech_source()):
                        await self._send_event(event)

                if not has_speech:
                    await self.websocket.send_json({"type": "silence"})

        except WebSocketDisconnect:
            logger.info("ws_session_disconnected")
        except Exception as e:
            logger.error("ws_session_error", error=str(e))
            try:
                await self.websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    async def _send_event(self, event: Any) -> None:
        """Send a pipeline event to the client via WebSocket."""

        if event.type == "transcript":
            await self.websocket.send_json({
                "type": "transcript",
                "text": event.data,
                "is_final": False,
                "latency_ms": event.latency_ms,
            })
        elif event.type == "audio":
            wav_bytes = to_wav_bytes(event.data, 24000)
            await self.websocket.send_bytes(wav_bytes)
        elif event.type == "error":
            await self.websocket.send_json({
                "type": "error",
                "message": str(event.data),
            })
