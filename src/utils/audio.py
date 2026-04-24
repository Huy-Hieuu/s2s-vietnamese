"""Audio utilities: resampling, format conversion, chunking."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import numpy as np
from numpy.typing import NDArray


def to_float32(audio: NDArray[np.int16 | np.float32]) -> NDArray[np.float32]:
    """Convert int16 PCM or any float array to float32 in [-1, 1]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def to_int16(audio: NDArray[np.float32]) -> NDArray[np.int16]:
    """Convert float32 audio to int16 PCM."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def resample(
    audio: NDArray[np.float32],
    orig_sr: int,
    target_sr: int,
) -> NDArray[np.float32]:
    """Resample audio to a new sample rate using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    new_length = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def to_wav_bytes(
    audio: NDArray[np.float32],
    sample_rate: int,
) -> bytes:
    """Encode float32 audio as 16-bit PCM WAV bytes."""
    import wave

    pcm = to_int16(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def chunk_audio(
    audio: NDArray[np.float32],
    chunk_size: int,
    hop: int | None = None,
) -> list[NDArray[np.float32]]:
    """Split audio into overlapping chunks.

    Args:
        audio: 1D float32 array.
        chunk_size: Samples per chunk.
        hop: Stride between chunk starts. Defaults to chunk_size (no overlap).

    Returns:
        List of float32 arrays, each of length chunk_size (last may be shorter).
    """
    if hop is None:
        hop = chunk_size
    chunks = []
    for start in range(0, len(audio), hop):
        end = min(start + chunk_size, len(audio))
        chunks.append(audio[start:end])
    return chunks


async def async_chunk_reader(
    audio: NDArray[np.float32],
    chunk_size: int,
) -> AsyncIterator[NDArray[np.float32]]:
    """Yield audio chunks as an async iterator."""
    for i in range(0, len(audio), chunk_size):
        yield audio[i : i + chunk_size]
