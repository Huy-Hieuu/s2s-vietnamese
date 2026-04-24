"""TTS configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str = "FunAudioLLM/CosyVoice2-0.5B"
    sample_rate: int = 24000
    n_mel_channels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    torch_dtype: str = "bfloat16"


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "checkpoints/tts/cosyvoice2-vi"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 5
    logging_steps: int = 50


@dataclass(frozen=True)
class DataConfig:
    train_files: list[str] = field(default_factory=list)
    eval_files: list[str] = field(default_factory=list)
    min_audio_length_s: float = 1.0
    max_audio_length_s: float = 15.0
    sample_rate: int = 24000


@dataclass(frozen=True)
class VoiceCloningConfig:
    enabled: bool = True
    reference_audio_dir: str = "data/tts/reference_speakers"
    min_reference_duration_s: float = 3.0
    max_speakers: int = 100


@dataclass(frozen=True)
class StreamingConfig:
    chunk_size_tokens: int = 50
    overlap_tokens: int = 5


@dataclass(frozen=True)
class TTSConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    voice_cloning: VoiceCloningConfig = field(default_factory=VoiceCloningConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


def load_tts_config(path: str | Path) -> TTSConfig:
    """Load TTS config from a YAML file."""
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return TTSConfig(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        voice_cloning=VoiceCloningConfig(**raw.get("voice_cloning", {})),
        streaming=StreamingConfig(**raw.get("streaming", {})),
    )
