"""ASR fine-tuning entrypoint for Whisper on Vietnamese data."""

from __future__ import annotations

import argparse

from src.asr.config import load_asr_config
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Vietnamese ASR")
    parser.add_argument("--config", type=str, required=True, help="Path to ASR config YAML")
    return parser.parse_args()


def train(config_path: str) -> None:
    """Run ASR fine-tuning with the given config."""
    config = load_asr_config(config_path)
    logger.info("starting_asr_training", model=config.model.name, output=config.training.output_dir)

    import evaluate
    from datasets import load_dataset
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    processor = AutoProcessor.from_pretrained(config.model.name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(config.model.name)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred: object) -> dict[str, float]:
        pred_ids = getattr(pred, "predictions", pred)
        label_ids = getattr(pred, "label_ids", None)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        logging_steps=config.training.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
    )

    # Dataset loading placeholder — replace with actual data pipeline
    train_dataset = load_dataset("common_voice", "vi", split="train")
    eval_dataset = load_dataset("common_voice", "vi", split="test")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    logger.info("asr_training_complete", output=config.training.output_dir)


def main() -> None:
    setup_logging()
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
