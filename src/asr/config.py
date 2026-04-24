"""ASR configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "openai/whisper-large-v3"
    language: str = "vi"
    use_flash_attention_2: bool = True
    torch_dtype: str = "bfloat16"


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "checkpoints/asr/whisper-vi"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 50


@dataclass(frozen=True)
class DataConfig:
    train_files: list[str] = field(default_factory=list)
    eval_files: list[str] = field(default_factory=list)
    accent_distribution: dict[str, float] = field(
        default_factory=lambda: {"bac": 0.4, "trung": 0.3, "nam": 0.3}
    )
    max_audio_length_s: float = 30.0
    sampling_rate: int = 16000


@dataclass(frozen=True)
class StreamingConfig:
    chunk_size_s: float = 3.0
    stride_s: float = 1.0
    language: str = "vi"
    task: str = "transcribe"


@dataclass(frozen=True)
class ASRConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


def load_asr_config(path: str | Path) -> ASRConfig:
    """Load ASR config from a YAML file."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return ASRConfig(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        streaming=StreamingConfig(**raw.get("streaming", {})),
    )
