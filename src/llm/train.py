"""LLM training — continued pretraining and SFT for Vietnamese."""

from __future__ import annotations

import argparse

from src.llm.config import load_llm_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLM for Vietnamese")
    parser.add_argument("--config", type=str, required=True, help="Path to LLM config YAML")
    parser.add_argument(
        "--mode",
        choices=["pretrain", "sft"],
        default="sft",
        help="Training mode",
    )
    return parser.parse_args()


def train(config_path: str, mode: str = "sft") -> None:
    """Run LLM training with the given config."""
    config = load_llm_config(config_path)
    logger.info(
        "starting_llm_training",
        mode=mode,
        model=config.model.name,
        output=config.training.output_dir,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Verify Vietnamese diacritics handling
    test_text = "Xin chào, tôi là trợ lý AI."
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    if decoded.strip() != test_text:
        logger.warning("tokenizer_diacritics_mismatch", original=test_text, decoded=decoded)

    _model = AutoModelForCausalLM.from_pretrained(config.model.name)

    _training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        logging_steps=config.training.logging_steps,
    )

    # Dataset loading placeholder — replace with actual data pipeline
    logger.info("llm_training_placeholder", note="Replace with actual dataset loading")

    logger.info("llm_training_complete", output=config.training.output_dir)


def main() -> None:
    setup_logging()
    args = parse_args()
    train(args.config, args.mode)


if __name__ == "__main__":
    main()
