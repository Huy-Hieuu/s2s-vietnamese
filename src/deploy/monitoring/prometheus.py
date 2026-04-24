"""Prometheus metrics for the S2S pipeline."""

from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Metrics are defined lazily to avoid import errors when prometheus_client
# is not installed in development environments.

_METRICS_DEFINED = False

# Pipeline metrics
PIPELINE_LATENCY: Any = None
PIPELINE_REQUESTS: Any = None
PIPELINE_ERRORS: Any = None

# Module metrics
ASR_LATENCY: Any = None
LLM_LATENCY: Any = None
TTS_LATENCY: Any = None
GPU_UTILIZATION: Any = None


def _define_metrics() -> None:
    """Define all Prometheus metrics."""
    global _METRICS_DEFINED, PIPELINE_LATENCY, PIPELINE_REQUESTS, PIPELINE_ERRORS
    global ASR_LATENCY, LLM_LATENCY, TTS_LATENCY, GPU_UTILIZATION

    if _METRICS_DEFINED:
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        PIPELINE_LATENCY = Histogram(
            "s2s_pipeline_latency_seconds",
            "End-to-end pipeline latency",
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
        )
        PIPELINE_REQUESTS = Counter(
            "s2s_pipeline_requests_total",
            "Total pipeline requests",
        )
        PIPELINE_ERRORS = Counter(
            "s2s_pipeline_errors_total",
            "Total pipeline errors",
            ["stage"],
        )
        ASR_LATENCY = Histogram(
            "s2s_asr_latency_seconds",
            "ASR transcription latency",
        )
        LLM_LATENCY = Histogram(
            "s2s_llm_latency_seconds",
            "LLM generation latency",
        )
        TTS_LATENCY = Histogram(
            "s2s_tts_latency_seconds",
            "TTS synthesis latency",
        )
        GPU_UTILIZATION = Gauge(
            "s2s_gpu_utilization_percent",
            "GPU utilization percentage",
        )
        _METRICS_DEFINED = True
    except ImportError:
        logger.warning("prometheus_client_not_installed")


def setup_metrics(app: Any) -> None:
    """Attach Prometheus metrics to the FastAPI app."""
    _define_metrics()
    try:
        from prometheus_client import make_asgi_app

        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)
        logger.info("prometheus_metrics_mounted")
    except ImportError:
        logger.warning("prometheus_not_available")
