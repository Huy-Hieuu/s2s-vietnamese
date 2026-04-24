# End-to-End Speech Models (Phase 2)

## What is E2E Speech?

End-to-end speech models skip the text intermediate entirely. Instead of Audio → Text → Text → Audio, they go:

```
Audio → Speech Tokens → Multimodal LLM → Speech Tokens → Audio
```

This preserves information that text can't represent:
- **Tone of voice** — sarcastic, excited, hesitant
- **Emotion** — happy, sad, frustrated
- **Non-speech sounds** — laughter, breathing, pauses
- **Speaker identity** — voice characteristics, accent

## Why E2E?

The cascaded pipeline has a fundamental limitation: converting speech to text loses information. Consider:

```
User says (enthusiastically): "THAT'S AMAZING!"
ASR output (flat): "That's amazing."
LLM responds: "I'm glad you're excited!"
TTS speaks: (flat voice) "I'm glad you're excited."
```

The enthusiasm is lost at the ASR stage. An E2E model could preserve it:

```
User audio: [enthusiastic tone, raised pitch]
E2E model: → [enthusiastic response, matching energy]
```

## Speech Tokenization

### What Are Speech Tokens?

Just as text is tokenized into discrete integers (word pieces, BPE tokens), audio can be tokenized into discrete codes:

```
Raw Audio Waveform (24kHz, continuous)
    ↓
Speech Encoder (EnCodec or DAC)
    ↓
Discrete Tokens: [[45, 123, 789, ...],   ← codebook 1
                  [12, 456, 234, ...],    ← codebook 2
                  [89, 345, 567, ...],    ← codebook 3
                  ...]                     ← codebook N
```

Each codebook captures a different aspect of the audio:
- **Codebook 1:** Broad phonetic content (what is being said)
- **Codebook 2-4:** Speaker identity, pitch contour
- **Codebook 5-8:** Fine acoustic details, prosody

### EnCodec

[Meta's EnCodec](https://arxiv.org/abs/2210.03087) is a neural audio codec that compresses audio into discrete tokens:

**Architecture:**
```
Audio (24kHz) → Encoder → Quantizer (RVQ) → Tokens
                                              ↓
Audio (24kHz) ← Decoder ← Dequantizer   ← Tokens
```

- **Encoder:** Strided convolutional network that compresses audio by 320× (75 tokens per second at 24kHz)
- **RVQ (Residual Vector Quantization):** Multiple codebooks that progressively refine the representation
- **Decoder:** Reconstructs audio from quantized tokens

**Key parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| Codebook size | 1024 | 1024 entries per codebook (10 bits) |
| Num codebooks | 8 | 8 levels of refinement |
| Bandwidth | 6.0 kbps | Compression rate |
| Frame rate | 75 Hz | 75 token frames per second |

### DAC (Descript Audio Codec)

An alternative to EnCodec with higher quality reconstruction. DAC uses:
- More codebooks (9 vs 8)
- Larger codebook size (4096 vs 1024)
- Better quantizer dropout during training

Trade-off: Higher quality but more tokens per second → more compute for the LLM.

## Multimodal LLM Architecture

### How It Works

The multimodal LLM processes interleaved text and audio tokens:

```
Input sequence:
[<text_tok>] [<text_tok>] ... [<audio_c1>] [<audio_c2>] ... [<text_tok>] ...
     ↓                 ↓            ↓              ↓               ↓
Embedding Layer (shared for text + audio codebooks)
     ↓
Transformer LLM (standard architecture with extended vocabulary)
     ↓
Output: [<text_tok>] or [<audio_codebook_1>] ... [<audio_codebook_8>]
```

### Training Approach

```
Stage 1: Speech Tokenizer Training
  - Train EnCodec/DAC on Vietnamese speech
  - Target: high-quality reconstruction at target bandwidth

Stage 2: Modality Alignment
  - Pre-train the LLM to understand speech tokens
  - Tasks: speech-to-text, text-to-speech, speech-to-speech
  - Teaches the LLM that audio tokens represent the same concepts as text

Stage 3: Instruction Fine-Tuning
  - Train on voice conversation data
  - Input: user audio tokens, Output: response audio tokens
  - Teaches conversational behavior with preserved prosody
```

### Challenges for Vietnamese

1. **Tonal preservation** — the speech tokenizer must encode tone information in its codebooks. EnCodec trained only on English may lose tonal detail. Fine-tuning on Vietnamese audio is essential.

2. **Token vocabulary size** — with 8 codebooks × 1024 entries, the audio vocabulary adds 8192 tokens. Combined with text vocabulary (~128K for Llama-3), the embedding table becomes very large.

3. **Training data** — E2E models need paired audio conversations, not just text. This data is much harder to collect than text-only data.

4. **Evaluation** — no standardized benchmarks for Vietnamese E2E speech. Need to build custom evaluation sets.

## Latency Comparison: Cascaded vs E2E

| Aspect | Cascaded | E2E |
|--------|----------|-----|
| ASR/Encoder | 200-500ms | 50-100ms (encoder only) |
| LLM inference | 100-300ms | 200-500ms (larger context) |
| TTS/Decoder | 100-300ms | 50-100ms (decoder only) |
| **Total** | **400-1100ms** | **300-700ms** |
| Quality | Text bottleneck | Preserves prosody |

The E2E model has the potential for lower latency because:
- No text intermediate → one less model to run
- Speech encoder is faster than full ASR (no autoregressive decoding)
- Audio decoder is faster than full TTS (parallel generation possible)

But the LLM context is longer (audio tokens are more numerous than text tokens), which can slow down inference.

## When to Switch from Cascaded to E2E

Consider E2E when:
- The cascaded pipeline quality plateaus (ASR errors are the bottleneck)
- You need to preserve emotion/prosody in responses
- You have 100+ hours of paired Vietnamese conversation data
- The cascaded latency can't meet the <1s target

Stay with cascaded when:
- You need to ship quickly with proven models
- Text-based quality is acceptable for your use case
- Training data is limited
- You need to debug and inspect intermediate outputs
