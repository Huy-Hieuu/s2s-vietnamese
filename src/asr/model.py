"""Whisper model wrapper for Vietnamese ASR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.asr.config import ASRConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WhisperWrapper:
    """Wraps a HuggingFace Whisper model for transcription.

    Handles model loading, dtype configuration, and Flash Attention 2
    setup for H100 inference.
    """

    def __init__(self, config: ASRConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """Load model and processor from HuggingFace or local checkpoint."""
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_name = checkpoint_path or self.config.model.name
        dtype = getattr(torch, self.config.model.torch_dtype)

        logger.info("loading_asr_model", model=model_name, dtype=self.config.model.torch_dtype)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": self.device,
        }
        if self.config.model.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **model_kwargs)
        self._processor = AutoProcessor.from_pretrained(model_name)
        logger.info("asr_model_loaded")

    def transcribe(
        self,
        audio: NDArray[np.float32],
        sampling_rate: int = 16000,
    ) -> str:
        """Transcribe audio array to text.

        Args:
            audio: Float32 audio array, mono, in [-1, 1].
            sampling_rate: Sample rate of the input audio.

        Returns:
            Transcribed text string.
        """
        import torch

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self._processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                language=self.config.streaming.language,
                task=self.config.streaming.task,
            )

        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
