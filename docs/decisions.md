# Architecture Decision Records

## ADR-001: Cascaded Pipeline Architecture

**Date:** 2026-04-22
**Status:** Accepted

### Context
We need a system that converts Vietnamese speech input to speech output with low latency. The system must be trainable end-to-end and deployable on a single H100 GPU. Two approaches were considered: a cascaded pipeline (ASR → LLM → TTS) and an end-to-end multimodal model.

### Options

1. **Cascaded pipeline (ASR → LLM → TTS)** — Use established models for each stage, wire them with streaming async generators.
2. **End-to-end multimodal model** — Single model that processes audio tokens directly, outputting audio tokens.
3. **Hybrid** — Start with cascaded, migrate to E2E later.

### Decision

We chose option 3 (hybrid). Phase 1 implements the cascaded pipeline for rapid development using proven components (Whisper, Llama-3, CosyVoice2). Phase 2 will explore end-to-end with a speech tokenizer + multimodal LLM.

**Rationale:**
- Cascaded allows independent training and evaluation of each module
- Established models have better Vietnamese support out of the box
- Streaming is easier to implement per-module than end-to-end
- End-to-end model training requires more research and data

### Consequences

- Each module can be optimized independently
- Latency is additive (ASR + LLM + TTS) but streaming overlaps stages
- Error propagation between modules needs monitoring
- Phase 2 E2E model will need a comparison baseline from Phase 1

---

## ADR-002: BF16 over FP16

**Date:** 2026-04-24
**Status:** Accepted

### Context
All models need a floating-point format for training and inference on H100.

### Decision
Use bf16 (bfloat16) everywhere — no fp16.

### Rationale
- H100 has native bf16 hardware support
- bf16 has the same dynamic range as fp32 (8-bit exponent) — no overflow/underflow
- fp16 requires loss scaling and can produce NaN gradients
- bf16 eliminates an entire class of training instability

---

## ADR-003: Async Generators for Streaming

**Date:** 2026-04-24
**Status:** Accepted

### Context
All pipeline stages must produce output incrementally for low-latency streaming.

### Decision
Use Python async generators (`async def` + `yield`) for all streaming interfaces. No threads, no queues, no blocking I/O.

### Rationale
- Single-threaded async is simpler to reason about than thread pools
- Async generators provide natural backpressure
- Composable — generators chain directly (ASR output → LLM input)
- FastAPI/WebSocket natively supports async

---

## ADR-004: vLLM for LLM Serving

**Date:** 2026-04-24
**Status:** Accepted

### Context
Need a production LLM inference engine that supports streaming and concurrent requests.

### Decision
Use vLLM's AsyncLLMEngine. Do not call the model directly.

### Rationale
- Continuous batching is essential for throughput under concurrent load
- PagedAttention prevents KV-cache memory fragmentation
- Prefix caching saves ~100ms per request on repeated system prompts
- Calling the model directly would process one request at a time
