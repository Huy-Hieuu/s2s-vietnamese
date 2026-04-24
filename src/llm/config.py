"""LLM configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    max_model_len: int = 4096
    torch_dtype: str = "bfloat16"
    use_flash_attention_2: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "checkpoints/llm/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 25


@dataclass(frozen=True)
class DataConfig:
    train_files: list[str] = field(default_factory=list)
    eval_files: list[str] = field(default_factory=list)
    max_seq_length: int = 4096
    chat_template: str = "llama-3"


@dataclass(frozen=True)
class ServingConfig:
    engine: str = "vllm"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    enable_prefix_caching: bool = True


@dataclass(frozen=True)
class LLMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)


def load_llm_config(path: str | Path) -> LLMConfig:
    """Load LLM config from a YAML file."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return LLMConfig(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        serving=ServingConfig(**raw.get("serving", {})),
    )
