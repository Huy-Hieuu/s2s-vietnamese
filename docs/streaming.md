# Streaming Pipeline Design

## Why Streaming?

In a voice conversation, **latency is everything**. If there's a noticeable gap between the user finishing speaking and the AI responding, the conversation feels unnatural. The target is < 1 second end-to-end — comparable to human response time in conversation.

Streaming is what makes this possible. Instead of waiting for each stage to complete before starting the next, we overlap computation:

```
Without streaming (batch):
User speaks ──────────────────────────────────> Response plays
              [VAD] → [ASR] → [LLM] → [TTS]     (3-5 seconds)

With streaming (overlapped):
User speaks ──────────────────────────> Response plays
              [VAD]
                   [ASR →→→]
                        [LLM →→→→→→]
                              [TTS →→→→→→→]
                                              (< 1 second perceived)
```

## Async Generators: The Core Abstraction

Every streaming interface in our pipeline is an **async generator** — a Python `async def` that `yield`s results incrementally:

```python
async def transcribe_stream(audio_iter: AsyncIterator) -> AsyncIterator[PartialTranscript]:
    async for chunk in audio_iter:
        # process chunk
        yield PartialTranscript(text="partial result")

    yield PartialTranscript(text="final result", is_final=True)
```

**Why async generators?**
- **Non-blocking** — the event loop can service multiple requests concurrently
- **Backpressure** — if the consumer is slow, the producer naturally slows down
- **Composable** — generators can be chained: ASR output feeds directly into LLM input
- **No threads** — everything runs on a single asyncio event loop per process

## The Three Streaming Interfaces

### 1. ASR Streaming (`src/asr/streaming.py`)

```python
async def transcribe_stream(
    audio_iter: AsyncIterator[NDArray[np.float32]]
) -> AsyncIterator[PartialTranscript]
```

- **Input:** Raw audio chunks from VAD
- **Output:** Partial and final transcripts
- **Strategy:** Sliding window — processes 3s audio windows with 1s stride
- **Key field:** `is_final` — signals when the utterance is complete

### 2. LLM Streaming (`src/llm/serve.py`)

```python
async def generate_stream(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> AsyncIterator[TokenChunk]
```

- **Input:** Formatted prompt string (system + user transcript)
- **Output:** Incremental text chunks as tokens are generated
- **Strategy:** vLLM's native streaming — yields each token as it's decoded
- **Key field:** `latency_ms` — tracks first-token latency (TTFT)

### 3. TTS Streaming (`src/tts/streaming.py`)

```python
async def synthesize_stream(
    text_iter: AsyncIterator[str],
    speaker_id: str = "default",
) -> AsyncIterator[AudioChunk]
```

- **Input:** Stream of text tokens from LLM
- **Strategy:** Buffers tokens until sentence boundary, then synthesizes
- **Output:** Audio chunks (float32 arrays) ready to play
- **Key field:** `sample_rate` — needed for playback

## Pipeline Wiring (`src/pipeline/cascade.py`)

The `CascadePipeline` composes the three streaming interfaces:

```python
class CascadePipeline:
    async def process(
        self,
        audio_iter: AsyncIterator[NDArray[np.float32]],
    ) -> AsyncIterator[PipelineEvent]:
        # Stage 1: ASR
        async for transcript in self._run_asr(audio_iter):
            yield PipelineEvent(type="transcript", data=transcript.text)
            if transcript.is_final:
                full_text = transcript.text
                break

        # Stage 2+3: LLM → TTS (overlapped)
        prompt = self._format_prompt(full_text)

        async def text_source():  # feeds LLM tokens to TTS
            async for chunk in self._llm.generate_stream(prompt):
                yield chunk.text

        async for audio_chunk in self._tts.synthesize_stream(text_source()):
            yield PipelineEvent(type="audio", data=audio_chunk.audio)
```

### Key Design Points

1. **The pipeline contains no ML logic** — it only wires modules together
2. **PipelineEvent** is a unified envelope: `{type, data, latency_ms, timestamp}`
3. **Latency tracking** — each stage reports its timing; the pipeline logs full breakdowns
4. **Error isolation** — a failure in one stage emits an error event, doesn't crash the pipeline

## Backpressure and Flow Control

Async generators naturally provide backpressure. If the WebSocket client is slow to consume audio chunks:

1. The TTS `synthesize_stream` pauses (can't yield)
2. The TTS buffer fills, but doesn't synthesize new chunks
3. The LLM `generate_stream` pauses (text_source isn't consumed)
4. vLLM buffers the output but doesn't OOM (PagedAttention manages KV cache)

This prevents memory blowup under load — the system self-throttles.

## WebSocket Protocol (`src/pipeline/realtime.py`)

### Client → Server

Binary frames containing raw PCM audio:
- Format: 16-bit signed integers, mono, 16kHz
- Continuous stream while user is speaking
- Client handles its own VAD (server VAD is a second opinion)

### Server → Client

Two frame types:

**JSON frames** (text events):
```json
{"type": "transcript", "text": "Xin chào", "is_final": false, "latency_ms": 234}
{"type": "silence"}
{"type": "error", "message": "..."}
```

**Binary frames** (audio):
- Raw WAV bytes (16-bit PCM, 24kHz, mono)
- Each frame is a complete audio chunk
- Client should queue and play sequentially

## Latency Optimization Strategies

### 1. Speculative ASR

Don't wait for the final transcript — start the LLM on partial transcripts:

```
ASR partial: "Xin chào"         → LLM starts with "Xin chào"
ASR partial: "Xin chào bạn"     → LLM context updated (or restart)
ASR final:   "Xin chào bạn nhé" → LLM has full context
```

Currently not implemented — our pipeline waits for the final transcript. Adding speculative ASR could save 200-500ms.

### 2. LLM Prefix Caching

The system prompt is the same for every request. vLLM's `enable_prefix_caching: true` pre-computes the KV cache for the system prompt, so each new request only needs to process the user's transcript.

Savings: **~100ms** per request (depends on system prompt length).

### 3. TTS Chunking

Smaller TTS chunks = audio starts playing sooner, but quality may decrease. The `chunk_size_tokens` parameter balances:
- **Small chunks (10-20 tokens):** Lower first-audio latency, but choppy prosody
- **Large chunks (50-100 tokens):** Better prosody, but delayed first audio
- **Our default (50 tokens):** Good balance for Vietnamese

### 4. Audio Pre-Roll

While TTS is synthesizing the first chunk, play a very short "thinking sound" (natural breathing or "ừm") to make the wait feel more human. This is a UX optimization, not a compute optimization.
