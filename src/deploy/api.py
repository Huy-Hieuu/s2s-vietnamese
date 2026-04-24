"""FastAPI application — main entry point for the S2S serving stack."""

from __future__ import annotations

import argparse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.asr.config import load_asr_config
from src.asr.model import WhisperWrapper
from src.deploy.monitoring.prometheus import setup_metrics
from src.llm.config import load_llm_config
from src.llm.serve import LLMServer
from src.pipeline.cascade import CascadePipeline
from src.pipeline.realtime import RealtimeSession
from src.pipeline.realtime import router as ws_router
from src.pipeline.vad import VoiceActivityDetector
from src.tts.config import load_tts_config
from src.tts.model import CosyVoiceWrapper
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load the pipeline configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle — startup and shutdown."""
    setup_logging(json_format=False)
    logger.info("s2s_server_starting")

    # Load configs
    pipeline_config = load_pipeline_config(
        app.state.config_path or "configs/pipeline/cascade.yaml"
    )

    # Initialize models
    asr_config = load_asr_config(pipeline_config["asr"]["config"])
    llm_config = load_llm_config(pipeline_config["llm"]["config"])
    tts_config = load_tts_config(pipeline_config["tts"]["config"])

    asr_model = WhisperWrapper(asr_config)
    asr_model.load()

    llm_server = LLMServer(llm_config)
    await llm_server.start()

    tts_model = CosyVoiceWrapper(tts_config)
    tts_model.load()

    vad_config = pipeline_config.get("vad", {})
    vad = VoiceActivityDetector(
        threshold=vad_config.get("threshold", 0.5),
        min_speech_duration_ms=vad_config.get("min_speech_duration_ms", 250),
        min_silence_duration_ms=vad_config.get("min_silence_duration_ms", 100),
        sampling_rate=vad_config.get("sampling_rate", 16000),
    )
    vad.load()

    pipeline = CascadePipeline(
        asr_model=asr_model,
        llm_server=llm_server,
        tts_model=tts_model,
        asr_config=asr_config,
        tts_config=tts_config,
        system_prompt=pipeline_config.get("llm", {}).get("system_prompt", ""),
        max_new_tokens=pipeline_config.get("llm", {}).get("max_new_tokens", 256),
        temperature=pipeline_config.get("llm", {}).get("temperature", 0.7),
        default_speaker=pipeline_config.get("tts", {}).get("default_speaker", "default"),
    )

    app.state.pipeline = pipeline
    app.state.vad = vad
    app.state.pipeline_config = pipeline_config

    logger.info("s2s_server_ready")
    yield

    # Shutdown
    await llm_server.stop()
    logger.info("s2s_server_stopped")


def create_app(config_path: str = "configs/pipeline/cascade.yaml") -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="S2S — Vietnamese Speech-to-Speech",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config_path = config_path

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics
    setup_metrics(app)

    # Routes
    app.include_router(ws_router, prefix="/ws")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    @app.websocket("/stream")
    async def websocket_stream(websocket: WebSocket) -> None:
        pipeline: CascadePipeline = websocket.app.state.pipeline
        vad: VoiceActivityDetector = websocket.app.state.vad
        session = RealtimeSession(websocket, pipeline, vad)
        await session.run()

    return app


app = create_app()


def main() -> None:
    """CLI entry point for s2s-serve."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the S2S serving stack")
    parser.add_argument("--config", default="configs/pipeline/cascade.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(
        "src.deploy.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
