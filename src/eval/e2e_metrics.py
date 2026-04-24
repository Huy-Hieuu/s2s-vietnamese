"""End-to-end evaluation — latency and quality metrics for the full pipeline."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Any

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class LatencyReport:
    """Latency percentile report for the pipeline."""

    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0


def compute_latency_percentiles(latencies: list[float]) -> LatencyReport:
    """Compute latency statistics from a list of measurements in ms."""
    if not latencies:
        return LatencyReport()
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return LatencyReport(
        p50_ms=sorted_lat[int(n * 0.50)],
        p95_ms=sorted_lat[int(n * 0.95)],
        p99_ms=sorted_lat[min(int(n * 0.99), n - 1)],
        mean_ms=statistics.mean(sorted_lat),
        min_ms=sorted_lat[0],
        max_ms=sorted_lat[-1],
    )


def evaluate_e2e(config_path: str) -> dict[str, Any]:
    """Run end-to-end pipeline evaluation.

    Measures latency percentiles and quality metrics across the full
    cascade pipeline under load.

    Returns:
        Dict with latency report and quality scores.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    e2e_config = config.get("e2e", {})
    targets = e2e_config.get("latency_targets", {})

    logger.info("evaluating_e2e", targets=targets)

    # Placeholder — run actual pipeline with test inputs
    latencies = [800.0, 850.0, 900.0, 950.0, 1000.0]
    report = compute_latency_percentiles(latencies)

    metrics = {
        "latency_p50_ms": report.p50_ms,
        "latency_p95_ms": report.p95_ms,
        "latency_p99_ms": report.p99_ms,
        "latency_mean_ms": report.mean_ms,
    }

    logger.info("e2e_evaluation_complete", **metrics)
    return metrics


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate end-to-end pipeline")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    metrics = evaluate_e2e(args.config)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.1f} ms")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
