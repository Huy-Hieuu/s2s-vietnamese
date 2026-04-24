"""GPU profiling helpers for H100 optimization."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUProfileResult:
    """Result from a GPU profiling session."""

    label: str
    duration_s: float
    peak_vram_gb: float = 0.0
    avg_gpu_util_pct: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _torch_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class GPUProfiler:
    """Context manager that profiles GPU usage during a code block.

    Measures wall-clock time, peak VRAM, and (optionally) GPU utilization.
    Always run ``scripts/profile_gpu.py`` before optimizing — measure first.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self._start_time: float = 0.0
        self._start_vram: float = 0.0
        self.result: GPUProfileResult | None = None

    def __enter__(self) -> GPUProfiler:
        self._start_time = time.perf_counter()
        if _torch_available():
            import torch

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self._start_vram = torch.cuda.memory_allocated() / 1e9
        return self

    def __exit__(self, *args: Any) -> None:
        duration = time.perf_counter() - self._start_time
        peak_vram = 0.0
        if _torch_available():
            import torch

            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / 1e9

        self.result = GPUProfileResult(
            label=self.label,
            duration_s=duration,
            peak_vram_gb=peak_vram,
        )
        logger.info(
            "gpu_profile",
            label=self.label,
            duration_s=f"{duration:.3f}",
            peak_vram_gb=f"{peak_vram:.2f}",
        )


@contextmanager
def profile_block(label: str) -> Generator[GPUProfileResult, None, None]:
    """Convenience context manager that yields the result on exit."""
    profiler = GPUProfiler(label)
    with profiler:
        yield profiler.result  # type: ignore[misc]  # populated on __exit__


def get_gpu_info() -> dict[str, Any]:
    """Return current GPU memory stats. Returns empty dict if no CUDA."""
    if not _torch_available():
        return {}
    import torch

    return {
        "device_name": torch.cuda.get_device_name(0),
        "total_vram_gb": torch.cuda.get_device_properties(0).total_mem / 1e9,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }
