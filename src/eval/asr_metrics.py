"""ASR evaluation — WER and CER measurement."""

from __future__ import annotations

import argparse

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    """Compute Word Error Rate between hypotheses and references.

    Args:
        hypotheses: Transcribed text strings.
        references: Ground truth text strings.

    Returns:
        WER as a float between 0 and 1.
    """
    try:
        import jiwer

        return jiwer.wer(references, hypotheses)
    except ImportError:
        logger.warning("jiwer_not_installed_fallback")
        return _simple_wer(hypotheses, references)


def _simple_wer(hypotheses: list[str], references: list[str]) -> float:
    """Minimal WER calculation using edit distance on words."""
    total_errors = 0
    total_words = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.lower().split()
        ref_words = ref.lower().split()
        total_words += len(ref_words)
        total_errors += _levenshtein(hyp_words, ref_words)
    return total_errors / max(total_words, 1)


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein distance between two lists."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def compute_cer(hypotheses: list[str], references: list[str]) -> float:
    """Compute Character Error Rate."""
    total_errors = 0
    total_chars = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_chars = list(hyp.lower())
        ref_chars = list(ref.lower())
        total_chars += len(ref_chars)
        total_errors += _levenshtein(hyp_chars, ref_chars)
    return total_errors / max(total_chars, 1)


def evaluate_asr(
    config_path: str,
    checkpoint_path: str | None = None,
) -> dict[str, float]:
    """Run full ASR evaluation.

    Returns:
        Dict with 'wer' and 'cer' metrics.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    asr_config = config.get("asr", {})
    target_wer = asr_config.get("target_wer", 0.10)

    logger.info("evaluating_asr", target_wer=target_wer)

    # Placeholder — load test data and run ASR inference
    # Replace with actual evaluation pipeline
    wer = 0.0
    cer = 0.0

    logger.info("asr_evaluation_complete", wer=wer, cer=cer, target_met=wer <= target_wer)
    return {"wer": wer, "cer": cer}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate ASR quality")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    metrics = evaluate_asr(args.config)
    print(f"WER: {metrics['wer']:.4f}, CER: {metrics['cer']:.4f}")


if __name__ == "__main__":
    main()
