# Large Language Model (LLM) — vLLM Serving

## What Does the LLM Do?

The LLM is the "brain" of the S2S pipeline. It receives the ASR transcript and generates a natural language response. For a voice assistant, this response should be:
- **Conversational** — natural spoken style, not formal written text
- **Concise** — people don't want to listen to long monologues
- **Contextually aware** — understands the conversation history
- **Vietnamese-native** — not translated from English

## Why Llama 3?

We use **Meta-Llama-3-8B-Instruct** as our base model:

- **8B parameters** — fits comfortably on H100 alongside ASR and TTS models
- **Strong multilingual ability** — handles Vietnamese well after fine-tuning
- **Instruction-tuned** — already knows how to follow conversational prompts
- **Efficient** — 8B is the sweet spot between quality and inference speed
- **Open weights** — can be fully fine-tuned and deployed without restrictions

### Why Not a Larger Model?

| Model | Params | VRAM (bf16) | Throughput | Vietnamese Quality |
|-------|--------|-------------|------------|-------------------|
| Llama-3-8B | 8B | ~16GB | High | Good (after SFT) |
| Llama-3-70B | 70B | ~140GB | Low | Excellent |
| Qwen-2-7B | 7B | ~14GB | High | Good |

70B doesn't fit on a single H100 (needs 2+ GPUs with tensor parallelism). 8B is the right tradeoff for single-GPU deployment.

## Training Pipeline

### Stage 1: Continued Pretraining

Goal: Improve Vietnamese language understanding on the base model.

```
Pre-trained Llama-3-8B
    ↓
+ Vietnamese text corpus (news, books, web text)
    ↓
Continued pretraining (same objective as original: next-token prediction)
    ↓
Model with stronger Vietnamese understanding
```

**Key considerations:**
- **Learning rate:** 5e-5 (higher than SFT — adapting to new language patterns)
- **Data volume:** ~1-5B tokens of Vietnamese text
- **Tokenizer check:** Verify Llama-3's tokenizer handles Vietnamese diacritics without splitting characters into bytes
- **Catastrophic forgetting:** Mix in 10-20% English data to retain English capability

### Stage 2: Supervised Fine-Tuning (SFT)

Goal: Teach the model to respond conversationally in Vietnamese.

```
Continued-pretrained model
    ↓
+ Vietnamese conversation dataset (instruction → response pairs)
    ↓
SFT training (maximize likelihood of correct responses)
    ↓
Conversational Vietnamese LLM
```

**Training data format (Llama-3 chat template):**

```xml
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

Bạn là trợ lý AI thông minh. Phản hồi tự nhiên bằng tiếng Việt.

<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Xin chào, hôm nay thời tiết thế nào?

<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Chào bạn! Tôi không có thông tin thời tiết theo thời gian thực, nhưng tôi có thể giúp bạn tìm hiểu dự báo thời tiết cho khu vực của bạn. Bạn ở đâu vậy?

<|eot_id|>
```

### Stage 3: DPO (Optional)

Direct Preference Optimization aligns the model with preferred response styles:

```
Prompt: "Kể cho tôi nghe một câu chuyện"

Chosen (preferred):  "Ngày xưa có một cô bé..." ← natural, conversational
Rejected:            "Tôi có thể kể cho bạn..." ← robotic, hedging
```

DPO trains the model to prefer the chosen style.

## Vietnamese Diacritics and Tokenization

### The Problem

Vietnamese uses a Latin-based alphabet (Quốc Ngữ) with extensive diacritics. A tokenizer that doesn't handle these properly will split words into meaningless fragments.

**Good tokenization:**
```
"chào" → ["chào"]                    # 1 token
"trợ lý" → ["trợ", " lý"]            # 2 tokens
```

**Bad tokenization (byte-level splitting):**
```
"chào" → ["ch", "à", "o"]            # 3 tokens — loses semantic meaning
```

### Verification

Our `src/llm/data.py` includes `verify_tokenizer_diacritics()` which round-trips test sentences:

```python
text = "Xin chào, tôi là trợ lý AI."
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
assert decoded.strip() == text  # Must round-trip perfectly
```

Llama-3's tokenizer (based on Tiktoken/BPE) handles Vietnamese diacritics correctly in most cases, but always verify.

## vLLM Serving

### Why vLLM?

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput inference engine for LLMs. It provides:

- **Continuous batching** — dynamically groups requests for maximum GPU utilization
- **PagedAttention** — efficient KV-cache management, avoiding memory fragmentation
- **Prefix caching** — caches the system prompt's KV-cache across requests
- **Async API** — native support for streaming responses via async generators

Without vLLM, a single concurrent request would block all others. With vLLM, we can handle 60+ concurrent streams on one H100.

### Continuous Batching

Traditional batching waits to fill a batch before processing:

```
Request 1: [wait..............] [process]
Request 2: [wait..............] [process]
Request 3:        [wait........] [process]
```

Continuous batching inserts new requests into running batches:

```
Request 1: [process → → → → → →]
Request 2: [process → → → → → →]
Request 3:    [process → → → → →]
Request 4:        [process → → →]
```

This dramatically reduces average latency under load.

### Key Serving Parameters

From `configs/llm/sft.yaml`:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `gpu_memory_utilization` | 0.85 | Use 85% of VRAM for KV cache (15% overhead) |
| `max_num_seqs` | 64 | Max concurrent request sequences |
| `max_num_batched_tokens` | 8192 | Max tokens processed per batch iteration |
| `enable_prefix_caching` | true | Cache system prompt KV across requests |
| `tensor_parallel_size` | 1 | Single GPU (H100 has enough VRAM for 8B) |

### Streaming Inference

```python
class LLMServer:
    async def generate_stream(self, prompt) -> AsyncIterator[TokenChunk]:
        async for request_output in engine.generate(prompt, ...):
            for output in request_output.outputs:
                new_text = output.text[len(full_text):]
                yield TokenChunk(text=new_text, ...)
```

Each yielded `TokenChunk` is a small piece of the response as it's generated. The TTS module starts synthesizing audio from these chunks immediately, without waiting for the full response.

## Prompt Engineering for Voice

The system prompt is critical for voice-appropriate responses:

```
Bạn là trợ lý AI thông minh. Phản hồi tự nhiên, ngắn gọn bằng tiếng Việt.
Bạn đang tham gia một cuộc trò chuyện bằng giọng nói, vì vậy hãy trả lời
như trong giao tiếp hàng ngày.
```

Key instructions:
- **"ngắn gọn"** (concise) — people don't want to listen to long responses
- **"giao tiếp hàng ngày"** (daily conversation) — spoken style, not written
- **"bằng tiếng Việt"** (in Vietnamese) — prevent English code-switching

## Memory Management

For an 8B model with bf16:
- **Model weights:** ~16GB
- **KV cache (per sequence, 4K tokens):** ~1GB
- **With vLLM at 85% utilization:** ~68GB available for KV cache
- **Max concurrent sequences:** ~60 (limited by KV cache, not model size)
