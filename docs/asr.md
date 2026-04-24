# Automatic Speech Recognition (ASR) — Whisper

## What is ASR?

ASR converts audio waveforms into text. In our pipeline, it's the second stage after VAD. The ASR receives speech audio segments and produces transcripts that the LLM uses to generate responses.

## Why Whisper?

[OpenAI's Whisper](https://arxiv.org/abs/2212.04356) is a state-of-the-art speech recognition model trained on 680,000 hours of multilingual audio. We chose it because:

- **Strong Vietnamese support** out of the box — trained on Vietnamese data
- **Robustness** — handles accents, background noise, and varied audio quality
- **Streaming-capable** — can produce partial transcripts from partial audio
- **Open source** — can be fine-tuned for domain-specific vocabulary
- **Flash Attention 2 compatible** — efficient on H100

## Whisper Architecture

### Encoder-Decoder Transformer

```
Audio Waveform (16kHz)
    ↓
Log-Mel Spectrogram (80 mel bins, 30s max)
    ↓
Encoder (32 transformer layers, d=1280)
    ↓   (encoded audio representation)
Decoder (32 transformer layers, d=1280)
    ↓
Token IDs → Text
```

**Encoder** converts the audio spectrogram into a sequence of hidden representations. Each encoder layer has:
- Multi-head self-attention (20 heads)
- Feed-forward network (d_ff = 4 × d_model)
- Layer normalization

**Decoder** is an autoregressive language model that generates text tokens one at a time, attending to both the encoder output and previously generated tokens.

### Model Sizes

| Size | Parameters | VRAM (bf16) | Relative WER |
|------|-----------|-------------|-------------|
| tiny | 39M | ~0.5GB | 3× worse |
| base | 74M | ~0.5GB | 2.5× worse |
| small | 244M | ~1GB | 2× worse |
| medium | 769M | ~2GB | 1.5× worse |
| **large-v3** | **1.55B** | **~3GB** | **1× (best)** |

We use **large-v3** — it fits easily in H100's 80GB VRAM alongside other models.

## Fine-Tuning for Vietnamese

### Why Fine-Tune?

Whisper's base Vietnamese performance is good (~15-20% WER) but not production-grade. Fine-tuning on curated Vietnamese data can push WER below 10%.

### Training Process

```
1. Load pre-trained Whisper large-v3
2. Freeze encoder (optional — preserves learned audio features)
3. Fine-tune decoder on Vietnamese audio-text pairs
4. Use LoRA or full fine-tuning depending on data size
```

### Data Requirements

| Accent | Target proportion | Rationale |
|--------|------------------|-----------|
| Bắc (Northern) | 40% | Largest speaker population |
| Trung (Central) | 30% | Most phonologically distinct |
| Nam (Southern) | 30% | Significant vocabulary differences |

Key data sources:
- [Common Voice Vietnamese](https://commonvoice.mozilla.org/) — crowdsourced, diverse accents
- [VIVOS](https://ailab.hcmus.edu.vn/vivos) — Vietnamese speech corpus
- Custom recorded conversations for domain-specific vocabulary

### Hyperparameters for Fine-Tuning

From `configs/asr/whisper_vietnamese.yaml`:

- **Learning rate:** 1e-5 (low — pre-trained model, small updates)
- **Batch size:** 16 per GPU × 2 gradient accumulation = effective 32
- **Epochs:** 3 (Whisper converges fast with good data)
- **bf16:** Yes — H100 has native bf16 support, no fp16 instability
- **Flash Attention 2:** Enabled — 2-4× faster attention on H100
- **Gradient checkpointing:** Yes — trades compute for VRAM savings

## Streaming ASR

### The Challenge

Standard Whisper processes a complete audio clip (up to 30 seconds). For real-time conversation, we need to start producing transcripts before the speaker finishes talking.

### Our Approach: Sliding Window

```
Audio stream: |----chunk1----|----chunk2----|----chunk3----|
                                     ↑ new audio arrives

Buffer:       [============audio buffer============]
              |← chunk_size (3s) →|
              |← stride (1s) →|

Transcripts:  "Xin chào"  →  "Xin chào, tôi"  →  "Xin chào, tôi là trợ lý"
              (partial)       (partial)            (final)
```

- **chunk_size_s = 3.0** — amount of audio processed per inference
- **stride_s = 1.0** — how far the window advances between inferences
- **Overlap** — 2 seconds of overlap between chunks stabilizes transcription

### StreamingASR Implementation

```python
class StreamingASR:
    async def transcribe_stream(self, audio_iter) -> AsyncIterator[PartialTranscript]:
        async for chunk in audio_iter:
            self._buffer += chunk              # accumulate audio
            while len(buffer) >= chunk_size:   # enough audio to process?
                text = model.transcribe(buffer[:chunk_size])
                yield PartialTranscript(text=text, is_final=False)
                buffer = buffer[stride:]       # advance window

        # Flush remaining audio
        if buffer:
            text = model.transcribe(buffer)
            yield PartialTranscript(text=text, is_final=True)
```

Key design choices:
- **Async generators** throughout — no blocking I/O, no threads
- **is_final flag** — tells downstream whether more audio is expected
- **Latency tracking** — each transcript includes its inference latency

## Vietnamese-Specific Challenges

### Tonal Language

Vietnamese has 6 tones that change word meaning:

| Tone | Diacritic | Example | Meaning |
|------|-----------|---------|---------|
| ngang (level) | (none) | ma | ghost |
| huyền (falling) | ` | mà | but |
| sắc (rising) | ´ | má | mother |
| hỏi (dipping-rising) | ̉ | mả | tomb |
| ngã (breaking-rising) | ~ | mã | horse |
| nặng (constricted) | . | mạ | rice seedling |

Whisper handles tones reasonably well because it was trained on Vietnamese audio, but fine-tuning significantly improves accuracy on tone-heavy vocabulary.

### Multi-Accent Recognition

Vietnamese accents differ in:
- **Pronunciation** — "v" is /v/ in North, /j/ in South
- **Diphthongs** — different vowel mergers across regions
- **Vocabulary** — "đồ ăn" (South) vs "thức ăn" (North) for "food"
- **Tone realization** — Northern tones are more distinct; Southern tones are closer together

Our training data mix (Bắc 40%, Trung 30%, Nam 30%) ensures the model handles all three major accents.

### Word Segmentation

Vietnamese is written without spaces between syllables, but most "words" are 1-2 syllables. Whisper's tokenizer handles this, but the LLM downstream may need segmented text for better understanding. Example:

- Raw: "tôi là trợ lý trí tuệ nhân tạo"
- Segmented: "tôi là trợ_lý trí_tuệ_nhân_tạo"

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **WER** (Word Error Rate) | % of words incorrectly transcribed | < 10% |
| **CER** (Character Error Rate) | % of characters incorrect (finer-grained) | < 5% |
| **Streaming latency** | Time from audio input to first transcript | < 500ms |

### WER Calculation

```
Reference: "xin chào tôi là trợ lý"
Hypothesis: "xin chào tôi là một trợ lý"

Substitutions: 0    Deletions: 0    Insertions: 1
WER = (S + D + I) / N = (0 + 0 + 1) / 7 = 14.3%
```
