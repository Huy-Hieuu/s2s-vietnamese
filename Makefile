.PHONY: install dev train-asr train-llm train-tts serve serve-dev \
        eval-asr eval-tts eval-e2e test lint format clean

PYTHON ?= python
PIP ?= pip

# ── Setup ────────────────────────────────────────────────────────────
install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev]"

# ── Training ─────────────────────────────────────────────────────────
train-asr:
	$(PYTHON) -m src.asr.train --config configs/asr/whisper_vietnamese.yaml

train-llm:
	$(PYTHON) -m src.llm.train --config configs/llm/sft.yaml

train-tts:
	$(PYTHON) -m src.tts.train --config configs/tts/cosyvoice2_vietnamese.yaml

# ── Serving ───────────────────────────────────────────────────────────
serve:
	$(PYTHON) -m src.deploy.api

serve-dev:
	$(PYTHON) -m uvicorn src.deploy.api:app --reload --host 0.0.0.0 --port 8000

# ── Evaluation ────────────────────────────────────────────────────────
eval-asr:
	$(PYTHON) -m src.eval.asr_metrics --config configs/eval/default.yaml

eval-tts:
	$(PYTHON) -m src.eval.tts_metrics --config configs/eval/default.yaml

eval-e2e:
	$(PYTHON) -m src.eval.e2e_metrics --config configs/eval/default.yaml

# ── Testing & Quality ─────────────────────────────────────────────────
test:
	pytest tests/ $(ARGS)

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

# ── Profiling ─────────────────────────────────────────────────────────
profile-asr:
	$(PYTHON) scripts/profile_gpu.py --module asr

profile-tts:
	$(PYTHON) scripts/profile_gpu.py --module tts

benchmark:
	$(PYTHON) scripts/benchmark_inference.py

# ── Docker ────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d

docker-logs:
	docker compose logs -f api

docker-down:
	docker compose down

# ── Cleanup ───────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info build dist .mypy_cache .pytest_cache .ruff_cache
