# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

Production-grade Vietnamese Speech-to-Speech (S2S) system on a single H100 80GB GPU. No cloud services — all training, inference, and deployment are self-hosted.

**Two-phase architecture:**
- **Phase 1 (Cascaded):** Microphone → VAD → ASR (Whisper) → LLM → TTS (CosyVoice2) → Speaker
- **Phase 2 (End-to-End):** Audio → Speech Tokenizer (EnCodec/DAC) → Multimodal LLM (text + audio tokens) → Audio output

## Common Commands

```bash
# Setup
pip install -e ".[dev]"

# Training
make train-asr        # Fine-tune Whisper for Vietnamese
make train-llm        # LLM continued pretraining + SFT
make train-tts        # Fine-tune CosyVoice2

# Serving
make serve            # Start FastAPI + vLLM serving stack
make serve-dev        # Dev mode with auto-reload

# Evaluation
make eval-asr         # WER measurement on Vietnamese test set
make eval-tts         # MOS, PESQ, speaker similarity
make eval-e2e         # End-to-end latency & quality

# Testing
pytest tests/                        # All tests
pytest tests/test_data_pipeline.py   # Single test file
pytest tests/ -k "test_vad"          # Single test by name
pytest tests/ -x --tb=short          # Fail fast

# Profiling
python scripts/profile_gpu.py --module asr   # GPU profiling (always before optimizing)
python scripts/benchmark_inference.py        # Throughput/latency benchmark

# Docker
docker compose up -d                         # Full stack (API + monitoring)
docker compose logs -f api                   # Follow API logs
```

## Architecture

### Module Boundaries

Each of the three core modules (ASR, LLM, TTS) is independently trainable, evaluatable, and servable. They are composed by `src/pipeline/cascade.py` into a streaming chain. The pipeline is the integration layer — it should contain no ML logic itself.

```
src/
├── asr/          # Whisper fine-tuning, streaming inference
├── llm/          # Pretraining, SFT, DPO, vLLM serving wrapper
├── tts/          # CosyVoice2/XTTS fine-tuning, streaming synthesis
├── pipeline/     # cascade.py wires ASR→LLM→TTS; vad.py; realtime.py (WebSocket)
├── e2e/          # Phase 2: speech tokenizer + multimodal LLM
├── eval/         # Per-module and end-to-end evaluation scripts
└── utils/        # audio.py, profiling.py, logging.py (no ML, pure utilities)
```

### Key Architectural Decisions

**Streaming-first design:** The cascaded pipeline uses async generators throughout — ASR emits partial transcripts, LLM streams tokens, TTS synthesizes chunk-by-chunk. Every module's `streaming.py` must yield audio chunks before the full output is ready. End-to-end target latency: < 1 second on H100.

**Config-driven:** No hardcoded hyperparameters. All configs live under `configs/{asr,llm,tts,pipeline}/` as YAML files. Use Hydra or plain YAML + dataclasses. Training scripts accept `--config path/to/config.yaml`.

**vLLM for LLM serving:** `src/llm/serve.py` wraps vLLM's `AsyncLLMEngine`. Do not bypass vLLM to call the model directly in production — continuous batching is critical for throughput.

**Vietnamese-specific concerns:**
- Tonal language: TTS must model pitch/F0 faithfully. Prefer models with explicit pitch conditioning.
- Multi-accent support: ASR training must include Bắc/Trung/Nam accent samples.
- Tokenization: verify that the LLM tokenizer handles Vietnamese diacritics without character splitting.

### Data Flow in Cascaded Pipeline

```
realtime.py (WebSocket)
  └─ vad.py           # Silero VAD — gates audio chunks to ASR
       └─ asr/streaming.py   # Whisper streaming → partial transcripts
            └─ llm/serve.py  # vLLM async streaming → token stream
                 └─ tts/streaming.py  # CosyVoice2 streaming → audio chunks
                      └─ WebSocket back to client
```

### Deployment Stack

- **FastAPI** — API layer with auth + rate limiting (`deploy/`)
- **vLLM** — LLM inference engine (separate process, called via async HTTP or Python API)
- **Prometheus + Grafana** — latency P50/P95/P99, GPU utilization, quality drift (`deploy/monitoring/`)
- **Docker Compose** — wires the full stack; `Dockerfile` is multi-stage

## Code Conventions

**Tensor shape annotations** (mandatory in all model code):
```python
# x: (batch, seq_len, d_model) → after projection: (batch, seq_len, n_heads, d_k)
```

**Logging:** Use structured logging from `src/utils/logging.py`, not `print()`.

**Function signatures:** Type hints required on all public functions. Google-style docstrings.

**Training runs:** Every training run must be logged to `docs/training_log.md` with config path, key metrics, and learnings.

**Architecture decisions:** Document in `docs/decisions.md` using the ADR format (ADR-{N}: Title / Date / Status / Context / Options / Decision / Consequences).

## H100 Usage Notes

- Default to `bf16` (not `fp16`) — H100 has native bf16 support
- Flash Attention 2 must be enabled for all transformer models
- DeepSpeed ZeRO-3 or FSDP for training runs that exceed 40GB
- Always run `scripts/profile_gpu.py` before any optimization — measure the actual bottleneck
- 80GB HBM3 allows large batch sizes; tune `per_device_train_batch_size` + gradient accumulation to maximize GPU utilization without OOM

## Phase Completion Checkpoints

| Phase | Key metric |
|-------|-----------|
| ASR | WER < 10% on Vietnamese test set; streaming latency < 500ms |
| LLM | Outperforms base model on Vietnamese benchmarks; profiling report written |
| TTS | MOS > 4.0; multi-speaker + voice cloning working; streaming |
| Pipeline | E2E latency < 1s on H100; stable under concurrent load (Locust) |
| Optimization | 2x+ throughput with < 1% quality degradation |
| Deployment | Production service running, monitored, auto-recoverable |
| E2E (Phase 2) | Working multimodal model with latency comparison vs cascaded |
