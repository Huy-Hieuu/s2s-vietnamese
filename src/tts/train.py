"""TTS fine-tuning entrypoint for CosyVoice2 on Vietnamese data."""

from __future__ import annotations

import argparse

from src.tts.config import load_tts_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CosyVoice2 for Vietnamese TTS")
    parser.add_argument("--config", type=str, required=True, help="Path to TTS config YAML")
    return parser.parse_args()


def train(config_path: str) -> None:
    """Run TTS fine-tuning with the given config."""
    config = load_tts_config(config_path)
    logger.info(
        "starting_tts_training",
        model=config.model.name,
        output=config.training.output_dir,
    )

    # Placeholder for actual CosyVoice2 fine-tuning
    # CosyVoice2 uses its own training loop — adapt to Trainer if needed
    logger.info("tts_training_placeholder", note="Implement CosyVoice2 fine-tuning pipeline")

    # Key considerations for Vietnamese TTS:
    # - Must model pitch/F0 faithfully for tonal accuracy
    # - Training data must cover Bắc/Trung/Nam accents
    # - Voice cloning requires reference audio per speaker

    logger.info("tts_training_complete", output=config.training.output_dir)


def main() -> None:
    setup_logging()
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
