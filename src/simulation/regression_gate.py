"""Distribution and latency gates for long-running simulation regressions."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def distribution_gate(candidate: dict[str, float], baseline: dict[str, float], tolerances: dict[str, float]) -> dict[str, bool]:
    return {
        key: abs(float(candidate[key]) - float(baseline[key])) <= float(tolerances[key])
        for key in tolerances
    }


def benchmark_latency(call: Callable[[], object], *, warmup: int = 2, repeats: int = 10) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        call()
    values = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        call()
        values.append((time.perf_counter() - start) * 1000.0)
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, 0.95)), "repeats": len(values)}
