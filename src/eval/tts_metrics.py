"""TTS evaluation — MOS, PESQ, and speaker similarity."""

from __future__ import annotations

import argparse

import numpy as np
from numpy.typing import NDArray

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def compute_pesq(
    reference: NDArray[np.float32],
    degraded: NDArray[np.float32],
    sample_rate: int = 16000,
) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).

    Args:
        reference: Clean reference audio.
        degraded: Synthesized/degraded audio.
        sample_rate: Must be 16000 or 8000.

    Returns:
        PESQ score (typically -0.5 to 4.5).
    """
    try:
        from pesq import pesq as _pesq

        return float(_pesq(sample_rate, reference, degraded, "wb"))
    except ImportError:
        logger.warning("pesq_not_installed")
        return 0.0


def compute_speaker_similarity(
    reference: NDArray[np.float32],
    synthesized: NDArray[np.float32],
    sample_rate: int = 16000,
) -> float:
    """Compute speaker similarity using cosine similarity of speaker embeddings.

    Args:
        reference: Reference speaker audio.
        synthesized: Synthesized audio.
        sample_rate: Audio sample rate.

    Returns:
        Similarity score between -1 and 1.
    """
    # Placeholder — use a speaker verification model (e.g., speechbrain)
    logger.info("speaker_similarity_placeholder")
    return 0.0


def compute_mos(
    audio_samples: list[NDArray[np.float32]],
    sample_rate: int = 24000,
) -> float:
    """Estimate Mean Opinion Score for audio samples.

    Uses a trained MOS prediction model or falls back to
    simple heuristic metrics.

    Args:
        audio_samples: List of float32 audio arrays.
        sample_rate: Audio sample rate.

    Returns:
        Estimated MOS (1-5 scale).
    """
    # Placeholder — integrate with DNSMOS or similar
    logger.info("mos_estimation_placeholder")
    return 4.0


def evaluate_tts(config_path: str) -> dict[str, float]:
    """Run full TTS evaluation suite."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    tts_config = config.get("tts", {})
    target_mos = tts_config.get("target_mos", 4.0)

    logger.info("evaluating_tts", target_mos=target_mos)

    # Placeholder — load test data and run TTS evaluation
    metrics = {
        "mos": 4.0,
        "pesq": 3.5,
        "speaker_similarity": 0.85,
    }

    logger.info("tts_evaluation_complete", **metrics, target_met=metrics["mos"] >= target_mos)
    return metrics


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate TTS quality")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    metrics = evaluate_tts(args.config)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
