# H100 GPU Optimization

## The H100 80GB

The NVIDIA H100 HBM3 is our target hardware. Key specs:

| Spec | Value |
|------|-------|
| HBM3 Memory | 80 GB |
| Memory bandwidth | 3.35 TB/s |
| BF16 tensor core performance | 989 TFLOPS |
| FP8 tensor core performance | 1,979 TFLOPS |
| TDP | 700W |
| NVLink bandwidth | 900 GB/s |

For S2S, the H100's strengths are:
- **80GB VRAM** — fits all three models (ASR ~3GB + LLM ~16GB + TTS ~4GB = ~23GB base, with KV cache taking the rest)
- **Native bf16** — no fp16 instability issues; bf16 has the same dynamic range as fp32
- **Flash Attention 2** — hardware-accelerated attention, 2-4× faster than standard attention

## Memory Planning

### Model VRAM Usage (bf16)

| Component | Parameters | VRAM (bf16) |
|-----------|-----------|-------------|
| Whisper large-v3 | 1.55B | ~3.1 GB |
| Llama-3-8B | 8B | ~16 GB |
| CosyVoice2-0.5B | 0.5B | ~1 GB |
| **Total model weights** | **~10B** | **~20 GB** |

### Runtime VRAM Usage

| Component | VRAM |
|-----------|------|
| Model weights | ~20 GB |
| vLLM KV cache (85% of remaining ~60GB) | ~51 GB |
| CUDA overhead, fragmentation | ~5 GB |
| ASR/TTS intermediate tensors | ~4 GB |
| **Total** | **~80 GB** |

This is tight but fits. If running OOM:
1. Reduce `gpu_memory_utilization` from 0.85 to 0.75
2. Reduce `max_model_len` from 4096 to 2048
3. Use model offloading for ASR/TTS (they're not used simultaneously with LLM)

## Optimization Techniques

### 1. BF16 (not FP16)

Always use `bf16` on H100. Unlike fp16:
- No overflow/underflow issues (same exponent range as fp32)
- No loss scaler needed
- No gradient instability
- H100 has native hardware support

```yaml
# In every config YAML
torch_dtype: "bfloat16"
bf16: true
fp16: false
```

### 2. Flash Attention 2

Flash Attention 2 is a hardware-aware attention implementation that:
- **Fuses** the QKV projection, attention, and output projection into one kernel
- **Avoids materializing** the full N×N attention matrix in HBM
- **Uses SRAM** (on-chip scratchpad) for intermediate computations

Performance impact:
| Model | Without Flash Attn | With Flash Attn 2 | Speedup |
|-------|--------------------|--------------------|---------|
| Whisper large-v3 | ~300ms / 5s audio | ~100ms | 3× |
| Llama-3-8B | ~50ms/token | ~15ms/token | 3.3× |

Enable in model configs:
```python
model_kwargs = {"attn_implementation": "flash_attention_2"}
```

**Requirement:** `pip install flash-attn` (compiles CUDA kernels, takes ~10 minutes)

### 3. Gradient Checkpointing

During training, trades compute for memory:
- **Without:** Store all intermediate activations → fast backward, high VRAM
- **With:** Recompute activations during backward → slow backward, low VRAM

```yaml
gradient_checkpointing: true
```

Saves ~60% of activation memory at ~25% compute overhead.

### 4. DeepSpeed ZeRO-3 / FSDP

For training runs that exceed single-GPU memory (e.g., Llama-70B fine-tuning):

**ZeRO-3** (DeepSpeed):
- Shards model weights, gradients, and optimizer states across GPUs
- Each GPU only holds 1/N of the model
- Requires CPU-GPU communication during forward/backward

**FSDP** (PyTorch native):
- Similar to ZeRO-3 but integrated into PyTorch
- Better compatibility with HuggingFace Transformers
- Preferred for new projects

For our single-H100 setup with 8B models, neither is needed. They become relevant when:
- Fine-tuning larger models (70B+)
- Batch sizes exceed available VRAM
- Running multiple experiments concurrently

### 5. Batch Size Tuning

The H100's 80GB allows large batch sizes. The optimal strategy:

```
per_device_train_batch_size × gradient_accumulation_steps = effective_batch_size
```

- **Start with** `per_device_train_batch_size = 4`, `gradient_accumulation = 4` → effective 16
- **Monitor GPU utilization** with `nvidia-smi` or `scripts/profile_gpu.py`
- **Increase batch_size** until GPU utilization > 90%
- **Don't OOM** — leave 10-15% VRAM headroom

### 6. CUDA Graphs

For inference, CUDA graphs capture the entire model execution as a single GPU operation:
- Eliminates kernel launch overhead
- Reduces CPU-GPU synchronization
- Best for fixed-shape inference (batch size, sequence length)

vLLM enables CUDA graphs automatically. For ASR/TTS, manual CUDA graph capture can help:

```python
# Pseudo-code
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(static_input)
# Replay: graph.replay()
```

## Profiling

**Always profile before optimizing.** Our `scripts/profile_gpu.py` measures:
- Model load time
- Inference latency
- Peak VRAM usage
- GPU utilization

```bash
# Profile ASR
python scripts/profile_gpu.py --module asr

# Profile TTS
python scripts/profile_gpu.py --module tts

# Benchmark throughput
python scripts/benchmark_inference.py --module asr --iterations 50
```

### Common Profiling Findings

| Bottleneck | Cause | Fix |
|------------|-------|-----|
| Low GPU utilization | Small batch size | Increase batch size |
| High VRAM, low utilization | Large model, small batch | Gradient checkpointing |
| High latency, low VRAM | Data loading bottleneck | Increase `num_workers` |
| OOM during training | Activation memory | Reduce batch size or enable gradient checkpointing |

## Model Loading Strategy

On a single H100, we need all three models loaded simultaneously. Memory allocation:

```python
# Order matters — load largest first to avoid fragmentation
1. LLM (16GB) → allocated in contiguous VRAM
2. ASR (3GB) → fits in remaining space
3. TTS (1GB) → fits easily
4. KV cache (51GB) → vLLM manages this dynamically
```

### Alternative: Sequential Loading

If VRAM is too tight, load models on-demand:

```python
# Only keep the LLM in VRAM permanently
# Load ASR model → transcribe → unload
# Load TTS model → synthesize → unload
```

This adds 2-3 seconds per request for model loading but saves ~4GB VRAM.
