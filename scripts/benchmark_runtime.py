from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "gfs-core" / "src"))
sys.path.insert(0, str(ROOT / "gfs-studio" / "src"))

from gfs_core.bootstrap.application import CoreApplication
from gfs_core.bootstrap.builtin_plugins import build_stable_registry
from gfs_core.kernel import PluginKernel
from gfs_core.kernel.project import source_identity
from gfs_studio.bootstrap.application import StudioApplication


def measure(operation):
    tracemalloc.start()
    started = time.perf_counter()
    try:
        value = operation()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        return value, {
            "elapsed_seconds": elapsed,
            "python_peak_bytes": peak,
        }
    finally:
        tracemalloc.stop()


def benchmark_replay(workspace: Path, stop_at_tick: int) -> dict:
    app = CoreApplication(ROOT / "gfs-core", workspace_root=workspace)
    expected, full = measure(lambda: app.run_micro_match(fast=True, seed=42))
    interrupted, partial = measure(lambda: app.run_micro_match(
        fast=True,
        seed=42,
        checkpoint="benchmark",
        stop_at_tick=stop_at_tick,
    ))
    resumed, resume = measure(lambda: app.run_micro_match(
        fast=True, seed=42, resume="benchmark",
    ))
    if resumed != expected:
        raise RuntimeError("benchmark resume did not converge")
    return {
        "uninterrupted": full,
        "interrupted": partial,
        "resumed": resume,
        "resume_prefix_ticks": interrupted["next_tick"],
        "resume_to_uninterrupted_ratio": resume["elapsed_seconds"] / full["elapsed_seconds"],
        "resume_matches_uninterrupted": True,
    }


def benchmark_tournament(workspace: Path) -> dict:
    registry = build_stable_registry(ROOT / "gfs-core", workspace_root=workspace)
    kernel = PluginKernel(registry)
    try:
        kernel.start(["gfs.core.tournament"])
        result, metrics = measure(kernel.context.require("tournament-runner").run)
        if result.get("champion") is None:
            raise RuntimeError("benchmark tournament did not complete")
        return {"tournament": metrics, "matches": 103, "champion": result["champion"]}
    finally:
        kernel.stop()


def benchmark_cohort(workspace: Path, stop_at_tick: int) -> dict:
    app = StudioApplication(ROOT / "gfs-studio")
    report, metrics = measure(lambda: app.export_cohort(
        [
            {"home": "Brazil", "away": "Argentina", "seed": 42},
            {"home": "France", "away": "England", "seed": 43},
        ],
        workspace,
        stop_at_tick=stop_at_tick,
    ))
    return {
        "cohort": metrics,
        "matches": len(report["matches"]),
        "rows": report["rows"],
    }


def main(arguments=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["replay", "tournament", "cohort", "all"], default="all")
    parser.add_argument("--stop-at-tick", type=int, default=60)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(arguments)
    with tempfile.TemporaryDirectory(prefix="gfs-benchmark-") as temporary:
        workspace = Path(temporary)
        results = {}
        if args.suite in {"replay", "all"}:
            results["replay"] = benchmark_replay(workspace / "replay", args.stop_at_tick)
        if args.suite in {"tournament", "all"}:
            results["tournament"] = benchmark_tournament(workspace / "tournament")
        if args.suite in {"cohort", "all"}:
            results["cohort"] = benchmark_cohort(workspace / "cohort", min(args.stop_at_tick, 2))
    report = {
        "schema_version": 1,
        "suite": args.suite,
        "source_identity": source_identity(ROOT),
        "results": results,
        "scope": "local_performance_not_production_sla",
        "memory_metric": "tracemalloc_python_peak_not_process_rss",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
