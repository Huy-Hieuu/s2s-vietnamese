"""Speech tokenizer — EnCodec/DAC for converting audio to discrete tokens."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SpeechTokenizer:
    """Wraps EnCodec or DAC for discrete audio tokenization.

    Converts raw audio into discrete token sequences suitable for
    multimodal LLM input. This is the front-end for Phase 2's
    end-to-end speech model.
    """

    def __init__(
        self,
        model_name: str = "facebook/encodec_24khz",
        device: str = "cuda",
        bandwidth: float = 6.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.bandwidth = bandwidth
        self._model: Any = None
        self._sample_rate: int = 24000

    def load(self) -> None:
        """Load the speech tokenizer model."""
        logger.info("loading_speech_tokenizer", model=self.model_name)
        try:
            from transformers import EncodecModel

            self._model = EncodecModel.from_pretrained(self.model_name).to(self.device)
            logger.info("speech_tokenizer_loaded")
        except ImportError:
            logger.warning("encodec_not_available")

    def encode(self, audio: NDArray[np.float32], sample_rate: int = 24000) -> NDArray[np.int64]:
        """Encode audio into discrete tokens.

        Args:
            audio: Float32 audio array.
            sample_rate: Sample rate of the audio.

        Returns:
            Token indices with shape (n_codebooks, n_frames).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self._model.encode(wav, bandwidth=self.bandwidth)
        return encoded.audio_codes.squeeze(0).cpu().numpy()

    def decode(self, tokens: NDArray[np.int64]) -> NDArray[np.float32]:
        """Decode tokens back to audio.

        Args:
            tokens: Token indices with shape (n_codebooks, n_frames).

        Returns:
            Float32 audio array.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        codes = torch.from_numpy(tokens).unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self._model.decode(codes, None)
        return decoded.squeeze().cpu().numpy()

    @property
    def codebook_size(self) -> int:
        """Number of entries per codebook."""
        if self._model is not None:
            return self._model.config.codebook_size
        return 1024

    @property
    def num_codebooks(self) -> int:
        """Number of codebooks (RVQ layers)."""
        if self._model is not None:
            return self._model.config.num_codebooks
        return 8
