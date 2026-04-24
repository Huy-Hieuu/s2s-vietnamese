# Deployment and Monitoring

## Deployment Architecture

```
                    ┌─────────────────────────────┐
                    │       H100 GPU Server        │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │   FastAPI (uvicorn)     │  │
                    │  │   :8000                 │  │
                    │  │                        │  │
                    │  │  /ws/stream  ──────┐   │  │
                    │  │  /health           │   │  │
                    │  │  /metrics          │   │  │
                    │  └────────────────────┼───┘  │
                    │                       │      │
                    │  ┌────────────────────▼───┐  │
                    │  │   Pipeline              │  │
                    │  │                        │  │
                    │  │  VAD → ASR → LLM → TTS │  │
                    │  │                        │  │
                    │  │  (all in one process)   │  │
                    │  └────────────────────────┘  │
                    │                              │
                    └──────────────────────────────┘
                                 │
                    ┌────────────┼─────────────┐
                    ▼            ▼              ▼
              Prometheus     Grafana        Client
               :9090         :3000      (WebSocket)
```

## FastAPI Application

The API layer (`src/deploy/api.py`) is the entry point. It handles:

### WebSocket Endpoint (`/ws/stream`)

Primary interface for real-time S2S:

```
Client                          Server
  │                               │
  │──── connect WS ──────────────→│
  │                               │
  │──── audio bytes ─────────────→│ VAD checks
  │                               │ ASR transcribes
  │                               │ LLM generates
  │                               │ TTS synthesizes
  │←─── JSON (transcript) ───────│
  │←─── binary (audio chunk) ────│
  │←─── binary (audio chunk) ────│
  │←─── JSON (transcript) ───────│
  │                               │
  │──── audio bytes ─────────────→│ (next utterance)
  │...                            │
```

### Health Check (`/health`)

```json
GET /health → {"status": "healthy"}
```

Used by Docker HEALTHCHECK and load balancers.

### Metrics (`/metrics`)

Prometheus-format metrics at `/metrics`:

```
s2s_pipeline_latency_seconds_bucket{le="0.5"} 42
s2s_pipeline_latency_seconds_bucket{le="1.0"} 89
s2s_pipeline_requests_total 95
s2s_asr_latency_seconds_sum 12.5
s2s_llm_latency_seconds_sum 8.3
s2s_tts_latency_seconds_sum 6.1
s2s_pipeline_errors_total{stage="asr"} 2
```

## Docker Setup

### Multi-Stage Dockerfile

```dockerfile
# Stage 1: Build (with CUDA compiler for flash-attn etc.)
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder
# Install dependencies, compile extensions

# Stage 2: Runtime (slim image, no compiler)
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
# Copy built packages, no build tools
```

Benefits:
- **Smaller image** — runtime image excludes compilers (~2GB savings)
- **Security** — fewer packages = smaller attack surface
- **Faster pulls** — smaller image deploys faster

### Docker Compose Stack

| Service | Purpose | Port |
|---------|---------|------|
| `api` | FastAPI S2S server | 8000 |
| `prometheus` | Metrics collection | 9090 |
| `grafana` | Dashboards and alerting | 3000 |

GPU access via Docker deploy config:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Monitoring

### Key Metrics to Track

| Metric | Type | Alert Threshold |
|--------|------|-----------------|
| `s2s_pipeline_latency_seconds` | Histogram | p95 > 1.5s |
| `s2s_pipeline_requests_total` | Counter | Rate change anomaly |
| `s2s_pipeline_errors_total` | Counter | Any increase |
| `s2s_asr_latency_seconds` | Histogram | p95 > 800ms |
| `s2s_llm_latency_seconds` | Histogram | p95 > 500ms |
| `s2s_tts_latency_seconds` | Histogram | p95 > 500ms |
| GPU utilization | Gauge | < 50% (waste) or > 95% (overload) |
| GPU VRAM usage | Gauge | > 90% (OOM risk) |

### Grafana Dashboard Layout

```
┌──────────────────────────────────────────────────────┐
│  S2S Pipeline Overview                                │
├───────────────┬───────────────┬──────────────────────┤
│  E2E Latency  │  Throughput   │  Error Rate          │
│  p50 p95 p99  │  req/s        │  by stage            │
├───────────────┼───────────────┼──────────────────────┤
│  ASR Latency  │  LLM Latency  │  TTS Latency         │
│  p50 p95      │  TTFT p95     │  First chunk p95     │
├───────────────┴───────────────┴──────────────────────┤
│  GPU Utilization  │  VRAM Usage  │  Temperature       │
└───────────────────┴──────────────┴────────────────────┘
```

### Quality Drift Detection

Monitor for quality degradation over time:
- **WER** — periodically run ASR evaluation on held-out test set
- **MOS** — sample TTS output and score (automated or human)
- **Response quality** — track user feedback signals

If quality drifts, it usually means:
- Input distribution changed (new accents, vocabulary)
- Model was accidentally overwritten
- Training data quality issue

## Rate Limiting

Using `slowapi` middleware:

```python
# Default: 60 requests/minute with burst of 10
limiter = Limiter(key_func=get_remote_address)
```

For a voice assistant, rate limiting prevents:
- Abuse (spamming the endpoint)
- GPU OOM (too many concurrent inference streams)
- Runaway costs (if using cloud GPU)

## Scaling Considerations

### Single H100 (current)

One server handles:
- **Concurrent streams:** ~60 (limited by vLLM KV cache)
- **Throughput:** ~20-30 conversations/second (depends on utterance length)

### Multi-GPU Scaling

For higher throughput:
- **Tensor parallelism:** Split one model across GPUs (lower latency per request)
- **Replication:** Run multiple pipeline instances, load balance (higher throughput)

### Auto-Recovery

The Docker setup includes `restart: unless-stopped` and health checks:
- If the API crashes → Docker restarts it
- If GPU OOM → model reloading on startup
- Prometheus alerts → notify on-call for persistent failures
