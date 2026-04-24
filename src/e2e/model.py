"""Multimodal LLM for end-to-end speech (Phase 2)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MultimodalOutput:
    """Output from the multimodal speech model."""

    text: str = ""
    audio_tokens: NDArray[np.int64] | None = None
    audio: NDArray[np.float32] | None = None


class MultimodalSpeechModel:
    """End-to-end multimodal model that processes both text and audio tokens.

    Phase 2 architecture: Audio → Speech Tokenizer → Multimodal LLM → Audio output.
    This replaces the cascaded pipeline with a single model.
    """

    def __init__(
        self,
        llm_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        speech_tokenizer_name: str = "facebook/encodec_24khz",
        device: str = "cuda",
    ) -> None:
        self.llm_name = llm_name
        self.speech_tokenizer_name = speech_tokenizer_name
        self.device = device
        self._model: object | None = None

    def load(self) -> None:
        """Load the multimodal model and speech tokenizer."""
        logger.info("loading_multimodal_model", llm=self.llm_name)
        # Phase 2 placeholder — implement when moving from cascaded to E2E
        logger.info("multimodal_model_placeholder")

    async def process_stream(
        self,
        audio: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> AsyncIterator[MultimodalOutput]:
        """Process audio input and stream multimodal output.

        Args:
            audio: Input audio float32 array.
            sample_rate: Sample rate of input audio.

        Yields:
            MultimodalOutput with incremental text and/or audio.
        """
        # Phase 2 placeholder
        yield MultimodalOutput(text="[E2E placeholder]")
