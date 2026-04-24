"""GPU profiling script — measure actual bottlenecks before optimizing.

Usage:
    python scripts/profile_gpu.py --module asr
    python scripts/profile_gpu.py --module tts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger, setup_logging
from src.utils.profiling import GPUProfiler, get_gpu_info

logger = get_logger(__name__)


def profile_asr(config_path: str = "configs/asr/whisper_vietnamese.yaml") -> dict:
    """Profile ASR module: model load time, inference latency, VRAM usage."""
    from src.asr.config import load_asr_config
    from src.asr.model import WhisperWrapper

    import numpy as np

    config = load_asr_config(config_path)
    model = WhisperWrapper(config)

    results = {}

    with GPUProfiler("asr_model_load") as p:
        model.load()
    results["model_load"] = {"duration_s": p.result.duration_s, "peak_vram_gb": p.result.peak_vram_gb}

    # Synthetic 5-second audio
    audio = np.random.randn(5 * config.data.sampling_rate).astype(np.float32) * 0.1

    with GPUProfiler("asr_inference_5s") as p:
        _ = model.transcribe(audio, config.data.sampling_rate)
    results["inference_5s"] = {"duration_s": p.result.duration_s, "peak_vram_gb": p.result.peak_vram_gb}

    return results


def profile_tts(config_path: str = "configs/tts/cosyvoice2_vietnamese.yaml") -> dict:
    """Profile TTS module."""
    from src.tts.config import load_tts_config
    from src.tts.model import CosyVoiceWrapper

    config = load_tts_config(config_path)
    model = CosyVoiceWrapper(config)

    results = {}

    with GPUProfiler("tts_model_load") as p:
        model.load()
    results["model_load"] = {"duration_s": p.result.duration_s, "peak_vram_gb": p.result.peak_vram_gb}

    return results


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Profile GPU usage for S2S modules")
    parser.add_argument("--module", choices=["asr", "tts", "all"], required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"GPU: {gpu_info['device_name']}")
        print(f"VRAM: {gpu_info['total_vram_gb']:.1f} GB")
    else:
        print("No CUDA GPU detected — profiling will be limited.")
        return

    all_results: dict = {"gpu": gpu_info}

    if args.module in ("asr", "all"):
        config = args.config or "configs/asr/whisper_vietnamese.yaml"
        all_results["asr"] = profile_asr(config)

    if args.module in ("tts", "all"):
        config = args.config or "configs/tts/cosyvoice2_vietnamese.yaml"
        all_results["tts"] = profile_tts(config)

    print("\n" + json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
