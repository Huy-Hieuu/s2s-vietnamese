"""TTS module — CosyVoice2 fine-tuning and streaming synthesis for Vietnamese."""

from src.tts.config import TTSConfig
from src.tts.model import CosyVoiceWrapper
from src.tts.streaming import StreamingTTS

__all__ = ["TTSConfig", "CosyVoiceWrapper", "StreamingTTS"]
