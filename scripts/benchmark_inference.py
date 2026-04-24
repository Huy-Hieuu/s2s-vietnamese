"""Inference benchmarking — throughput and latency measurement.

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --config configs/pipeline/cascade.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def benchmark_asr(
    num_iterations: int = 50,
    audio_duration_s: float = 5.0,
) -> dict:
    """Benchmark ASR throughput and latency."""
    import numpy as np
    from src.asr.config import load_asr_config
    from src.asr.model import WhisperWrapper

    config = load_asr_config("configs/asr/whisper_vietnamese.yaml")
    model = WhisperWrapper(config)
    model.load()

    audio = np.random.randn(int(audio_duration_s * 16000)).astype(np.float32) * 0.1

    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        model.transcribe(audio, 16000)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "module": "asr",
        "iterations": num_iterations,
        "audio_duration_s": audio_duration_s,
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "throughput_rps": num_iterations / (sum(latencies) / 1000),
    }


async def benchmark_pipeline(
    num_iterations: int = 20,
) -> dict:
    """Benchmark full cascade pipeline latency."""
    # Placeholder — requires all models loaded
    logger.info("pipeline_benchmark_placeholder")
    return {
        "module": "pipeline",
        "iterations": num_iterations,
        "note": "Requires all models loaded — run with real checkpoints",
    }


async def run_benchmarks(module: str, iterations: int) -> dict:
    """Run benchmarks for the specified module."""
    results = {}
    if module in ("asr", "all"):
        results["asr"] = await benchmark_asr(num_iterations=iterations)
    if module in ("pipeline", "all"):
        results["pipeline"] = await benchmark_pipeline(num_iterations=iterations)
    return results


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Benchmark S2S inference")
    parser.add_argument("--module", choices=["asr", "tts", "pipeline", "all"], default="all")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    results = asyncio.run(run_benchmarks(args.module, args.iterations))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
