# Voice Activity Detection (VAD)

## What is VAD?

Voice Activity Detection is the gatekeeper of the pipeline. Its job is to answer one question: **"Is someone speaking right now?"** It processes raw audio and segments it into speech chunks, filtering out silence and background noise before forwarding audio to the ASR module.

Without VAD, the ASR would constantly try to transcribe silence, wasting GPU compute and producing garbage output.

## Why VAD Matters

- **Compute savings** — ASR and downstream models only process actual speech
- **Latency reduction** — the pipeline doesn't wait for or process silence
- **Endpointing** — detecting when a speaker stops tells the system when to respond
- **User experience** — the system responds at natural conversation boundaries

## How Silero VAD Works

We use [Silero VAD](https://github.com/snakers4/silero-vad), a lightweight neural network that runs on CPU in real-time.

### Architecture

Silero VAD uses a small neural network (not a large transformer) optimized for edge deployment:

```
Audio Window (30ms)
    ↓
Feature Extraction (log-Mel spectrogram)
    ↓
Recurrent layers (GRU)
    ↓
Fully connected → sigmoid
    ↓
Speech probability [0.0 - 1.0]
```

- **Input:** 30ms audio windows at 16kHz (480 samples)
- **Output:** A single probability score between 0 and 1
- **Threshold:** Typically 0.5 (configurable). Above = speech, below = silence
- **Runtime:** ~1ms per window on CPU — negligible compared to ASR/LLM/TTS

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `threshold` | 0.5 | Higher = fewer false positives, but may miss quiet speech |
| `min_speech_duration_ms` | 250ms | Ignore speech shorter than this (filters coughs, clicks) |
| `min_silence_duration_ms` | 100ms | How long silence must last to end a speech segment |
| `window_size_ms` | 30ms | Audio window size for each prediction |
| `sampling_rate` | 16000 | Input audio sample rate |

### Segment Detection Algorithm

Our `VoiceActivityDetector` doesn't just classify individual windows — it tracks state across windows to produce clean speech segments:

```
1. Buffer incoming audio windows
2. For each window, get speech probability from Silero
3. If probability ≥ threshold → mark as speech, add to buffer
4. If probability < threshold and currently in speech → count silence
5. When silence duration ≥ min_silence_duration:
   a. If speech buffer ≥ min_speech_duration → emit SpeechSegment
   b. Clear buffer, reset state
6. At stream end → flush any remaining buffered speech
```

This stateful approach prevents:
- **Chattering** — rapidly alternating speech/silence on boundary windows
- **False triggers** — a single loud noise won't start a segment
- **Truncated speech** — short pauses within speech don't split the segment

## Integration in Our Pipeline

```python
# src/pipeline/vad.py
class VoiceActivityDetector:
    async def detect_speech(self, audio_stream) -> AsyncIterator[SpeechSegment]:
        # Processes audio_stream → yields SpeechSegment objects
```

The VAD sits between the WebSocket receiver and the ASR. Audio bytes come in, VAD gates them, and only speech segments reach the expensive ASR model.

### Energy-Based Fallback

When Silero VAD isn't available (no PyTorch, CPU-only dev environment), we fall back to a simple energy-based detector:

```python
energy = sqrt(mean(audio_window²))
probability = min(energy * 20.0, 1.0)
```

This is far less accurate than Silero but allows the system to function during development without GPU dependencies.

## Vietnamese-Specific Considerations

Vietnamese speech has characteristics that affect VAD tuning:

- **Tonal language** — some tones have very low energy at certain points (e.g., nặng tone drops sharply). A high threshold may cut off valid speech.
- **Short words** — Vietnamese has many single-syllable words. `min_speech_duration_ms` should be low enough (250ms) to capture these.
- **Conversational fillers** — "ờ", "à", "um" are common and should be detected as speech.

## Tuning VAD

1. Start with default threshold (0.5)
2. Record test conversations and check if all speech is captured
3. If speech is being cut off → lower threshold to 0.3-0.4
4. If background noise triggers VAD → raise threshold to 0.6-0.7
5. Monitor the `vad_ms` latency metric — should always be < 50ms
