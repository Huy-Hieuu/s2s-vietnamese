# Text-to-Speech (TTS) — CosyVoice2

## What Does TTS Do?

TTS converts text into natural-sounding speech audio. It's the final stage of the cascaded pipeline — the LLM generates text, and TTS turns that text into audio the user hears.

For a Vietnamese S2S system, TTS must handle:
- **Tones** — 6 distinct tones that change word meaning
- **Natural prosody** — rhythm, stress, intonation that sounds human
- **Multi-speaker support** — different voices for different use cases
- **Voice cloning** — replicate a specific speaker's voice from reference audio
- **Streaming** — produce audio chunks before the full text is ready

## Why CosyVoice2?

[CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) is a recent open-source TTS model from Alibaba with strong multilingual support:

- **Non-autoregressive architecture** — generates audio in parallel, faster than autoregressive models
- **Multi-speaker** — supports multiple voices out of the box
- **Voice cloning** — zero-shot cloning from 3+ seconds of reference audio
- **Streaming-capable** — can produce audio incrementally
- **Pitch conditioning** — explicit control over F0, critical for tonal languages

### Alternative: XTTS

[Coqui XTTS](https://github.com/coqui-ai/TTS) is another option with similar capabilities. We chose CosyVoice2 for its streaming performance and active development, but XTTS could be swapped in since the `StreamingTTS` interface is model-agnostic.

## TTS Architecture Concepts

### How Neural TTS Works

```
Text Input
    ↓
Text Encoder (converts characters/phonemes to embeddings)
    ↓
Duration Predictor (how long each phoneme should last)
    ↓
Acoustic Model (generates mel-spectrogram)
    ↓
Vocoder (converts spectrogram to audio waveform)
    ↓
Audio Output (24kHz float32)
```

### Key Components

**Text Encoder:** Converts Vietnamese text into a sequence of embeddings. For tonal languages, the encoder must be aware of tone markers (diacritics) since they change the phonetic realization.

**Duration Predictor:** Predicts how many mel-spectrogram frames each text token should span. Vietnamese tends to have more even durations than English (fewer stressed/unstressed contrasts), but tone realization varies in duration.

**Acoustic Model:** Generates the mel-spectrogram — a visual representation of the audio's frequency content over time:

```
Mel spectrogram (80 frequency bins × time frames)

Frequency ↑
  ████████████████████████  ← high frequencies (consonants, harmonics)
  ████████████████████████
  ████████████████████████
  ████████████████████████  ← low frequencies (vowels, fundamental)
  ████████████████████████
  └──────────────────────→ Time
```

**Vocoder:** Converts the mel-spectrogram into actual audio samples. Modern vocoders (HiFi-GAN, iSTFTNet) produce high-quality audio in real-time.

### Pitch/F0 and Vietnamese Tones

Vietnamese is a **tonal language** — the pitch contour of a syllable determines its meaning. This makes F0 (fundamental frequency) modeling critical:

| Tone | F0 Pattern | Example |
|------|-----------|---------|
| ngang (level) | Flat, mid-range | ma (ghost) |
| huyền (falling) | Starts high, falls | mà (but) |
| sắc (rising) | Starts mid, rises | má (mother) |
| hỏi (dipping-rising) | Dips then rises | mả (tomb) |
| ngã (breaking-rising) |Rises with glottalization | mã (horse) |
| nặng (constricted) | Falls with glottal stop | mạ (rice seedling) |

Without accurate F0 modeling, the TTS output would have wrong tones — making words incomprehensible. This is why we prefer models with **explicit pitch conditioning** (like CosyVoice2) over models that only implicitly learn pitch patterns.

## Streaming TTS

### The Challenge

Standard TTS synthesizes a complete utterance before producing any audio. For real-time conversation, we need to start playing audio before the LLM finishes generating its full response.

### Our Approach: Sentence-Boundary Streaming

```
LLM tokens: "Xin" " chào" " bạn" "." " Hôm" " nay" " tôi" " có" " thể" " giúp" " gì" "?" ...
                           ↑ sentence end        ↑ sentence end
                           ↓                     ↓
TTS chunk 1:  [audio: "Xin chào bạn."]
TTS chunk 2:  [audio: "Hôm nay tôi có thể giúp gì?"]
```

**How it works:**

1. Buffer incoming text tokens from the LLM
2. When a sentence boundary is detected (`.` `!` `?` or enough words), synthesize the buffer
3. Yield the resulting audio chunk immediately
4. Clear the buffer and continue

**Detection triggers:**
- Sentence-ending punctuation: `.`, `!`, `?`, `。`, `！`, `？`
- Buffer reaches `chunk_size_tokens` words without punctuation

### Why Not Phoneme-Level Streaming?

Synthesizing at the phoneme level would give lower latency but produces worse quality — the model needs sentence context for proper prosody and intonation. Sentence-boundary streaming is the standard tradeoff.

## Voice Cloning

### How It Works

CosyVoice2 supports zero-shot voice cloning from a short reference audio clip:

```
1. Record reference audio of target speaker (3-10 seconds)
2. Extract speaker embedding from reference audio
3. Condition the TTS on this embedding during synthesis
4. Output audio mimics the reference speaker's voice
```

### Configuration

From `configs/tts/cosyvoice2_vietnamese.yaml`:

```yaml
voice_cloning:
  enabled: true
  reference_audio_dir: "data/tts/reference_speakers"
  min_reference_duration_s: 3.0
  max_speakers: 100
```

- **min_reference_duration_s: 3.0** — shorter references produce poor cloning
- **max_speakers: 100** — pre-loaded speaker embeddings fit in memory

### Reference Audio Requirements

- **Duration:** 3-15 seconds of continuous speech
- **Quality:** Clean audio, no background noise, 24kHz preferred
- **Content:** Natural Vietnamese speech — the model doesn't need specific sentences
- **Diversity:** Include various tones and phonemes for better embedding quality

## Fine-Tuning for Vietnamese

### Training Data

Each training sample consists of:
- **Audio file** — Vietnamese speech at 24kHz, 1-15 seconds
- **Transcript** — Exact text of the spoken content
- **Speaker ID** — Identifier for voice consistency

**Data quality is critical** — misaligned transcripts produce garbled output. Verify:
- Audio matches transcript exactly (no paraphrasing)
- Consistent volume levels across samples
- No background music or noise
- Proper Vietnamese diacritics in transcripts

### Hyperparameters

From `configs/tts/cosyvoice2_vietnamese.yaml`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 1e-4 | Higher than ASR — TTS needs more adaptation |
| `batch_size` | 8 | Audio samples are variable-length |
| `epochs` | 10 | More epochs than ASR — smaller datasets |
| `max_audio_length_s` | 15 | Filter out overly long samples |
| `sample_rate` | 24000 | CosyVoice2 native rate |

## Evaluation Metrics

| Metric | What it measures | Target | How |
|--------|-----------------|--------|-----|
| **MOS** (Mean Opinion Score) | Perceived audio quality | > 4.0 (out of 5) | Human raters or DNSMOS |
| **PESQ** | Perceptual quality vs reference | > 3.5 | Automated comparison |
| **Speaker similarity** | How well cloning matches reference | > 0.85 | Cosine similarity of speaker embeddings |

### MOS Explained

MOS is the gold standard for TTS evaluation:

```
5 - Excellent (completely natural)
4 - Good (natural, minor artifacts)
3 - Fair (understandable, noticeable artifacts)
2 - Poor (difficult to understand)
1 - Bad (unintelligible)
```

Target: **> 4.0** means responses sound natural to Vietnamese speakers.
