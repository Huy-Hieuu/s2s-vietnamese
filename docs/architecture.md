# S2S Architecture Overview

## What is Speech-to-Speech?

Speech-to-Speech (S2S) is a system that takes spoken audio as input and produces spoken audio as output — essentially a real-time voice conversation with an AI. Unlike text chatbots, S2S must handle:

- **Audio preprocessing** — detecting when someone is speaking vs silence
- **Speech recognition** — converting audio waveforms to text
- **Language understanding & generation** — understanding intent and generating a response
- **Speech synthesis** — converting the response back to natural-sounding audio
- **Streaming** — doing all of this with minimal delay, before the full input/output is ready

## Two Approaches

### Cascaded Pipeline (Phase 1)

```
Microphone → VAD → ASR → LLM → TTS → Speaker
```

Each stage is a separate model, connected by async data streams. This is the approach we use in Phase 1.

**Advantages:**
- Each module can be trained, debugged, and swapped independently
- Uses proven, well-documented models (Whisper, Llama, CosyVoice2)
- Easier to add Vietnamese-specific optimization at each stage
- Streaming is straightforward — each stage emits partial results

**Disadvantages:**
- Latency is additive (each stage adds delay)
- Errors compound (ASR mistakes flow into LLM, which flows into TTS)
- Text is a lossy intermediate representation (tone of voice, emotion, pauses are lost)

### End-to-End Multimodal (Phase 2)

```
Audio → Speech Tokenizer → Multimodal LLM → Audio Output
```

A single model processes audio tokens directly and outputs audio tokens. No text intermediate.

**Advantages:**
- Preserves prosody, emotion, speaker identity through the pipeline
- Potentially lower latency (one model instead of three)
- Can handle non-speech sounds (laughter, breathing, hesitation)

**Disadvantages:**
- Requires significantly more training data and compute
- Less mature tooling and research
- Harder to debug (can't inspect intermediate text)
- Vietnamese-specific tuning is more complex

## Our Hybrid Strategy

We start with the cascaded pipeline because it works today with available models, and migrate to E2E as the research matures. The cascaded system also serves as a quality and latency baseline for evaluating the E2E model.

## System Data Flow

### Request Lifecycle

```
1. Client sends audio bytes via WebSocket
2. VAD checks if the audio contains speech
   ├─ No speech → send "silence" event, wait
   └─ Speech detected → forward to ASR
3. ASR produces partial transcripts as audio streams in
   └─ When speech ends → final transcript
4. LLM receives the transcript and streams response tokens
5. TTS converts streamed tokens to audio chunks as they arrive
6. Audio chunks are sent back to the client via WebSocket
```

### Streaming Overlap

The key to low latency is that stages overlap:

```
Time →
VAD:   ████████
ASR:          ████████████
LLM:                    ████████████████
TTS:                              ██████████████████
Audio:                                        ████████████████
```

- ASR starts before the full utterance is complete (partial transcripts)
- LLM starts generating as soon as it gets a partial transcript
- TTS starts synthesizing as soon as the LLM produces the first few tokens
- The client receives audio chunks while the LLM is still generating

### Latency Budget (target: < 1 second E2E)

| Stage | Target | Notes |
|-------|--------|-------|
| VAD | < 50ms | Lightweight model, runs per-window |
| ASR | < 500ms | Whisper streaming, partial transcripts |
| LLM first token | < 200ms | vLLM continuous batching, prefix caching |
| TTS first chunk | < 300ms | CosyVoice2 streaming synthesis |
| **E2E (perceived)** | **< 1000ms** | Overlapped streaming reduces wall-clock time |

## Module Independence

Each module in `src/` is independently:
- **Trainable** — `python -m src.asr.train --config ...`
- **Evaluatable** — `python -m src.eval.asr_metrics`
- **Servable** — each has its own config and can run standalone

The `pipeline/` module is pure wiring — it contains zero ML logic. This separation means you can upgrade the ASR model without touching the LLM or TTS code.

## Configuration System

All hyperparameters live in YAML files under `configs/`. No hardcoded values in source code. Each module's `config.py` defines frozen dataclasses that are loaded from YAML:

```python
# configs/asr/whisper_vietnamese.yaml → ASRConfig dataclass → passed to model
config = load_asr_config("configs/asr/whisper_vietnamese.yaml")
model = WhisperWrapper(config)
```

This makes it easy to:
- Run experiments with different hyperparameters without changing code
- Version control training configurations alongside results
- Reproduce any training run exactly
