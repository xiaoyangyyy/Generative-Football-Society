#!/usr/bin/env python3
"""Parallel, resumable matched-seed shadow across teams and fixtures."""

# This executable bootstraps the repository root before importing local packages.
# ruff: noqa: E402

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_engine.dataset_registry import write_json_atomic
from src.simulation.random_control import derive_seed, named_rng

TEAMS = (
    "Argentina",
    "Brazil",
    "England",
    "France",
    "Germany",
    "Japan",
    "Mexico",
    "Morocco",
    "Netherlands",
    "Portugal",
    "South Korea",
    "Spain",
    "United States",
    "Uruguay",
)
TARGETS = {"passes": 1070.54, "shots": 23.34, "xg": 2.6}
_WORLD = None


def fixtures(teams):
    teams = list(dict.fromkeys(teams))
    n = len(teams)
    if n < 4:
        raise ValueError("at least four distinct teams required")
    pairs = []
    for offset in (1, max(2, n // 2)):
        for i, team in enumerate(teams):
            pair = tuple(sorted((team, teams[(i + offset) % n])))
            if pair not in pairs:
                pairs.append(pair)
    return [oriented for pair in pairs for oriented in (pair, pair[::-1])]


def cfg(enabled, seconds):
    from src.match_engine.micro_config import MicroMatchConfig

    value = MicroMatchConfig.fast_demo()
    value.dt_default = 5.5
    value.match_seconds = seconds
    value.enable_subtick_reception_queue = enabled
    return value


def scalar(summary):
    passes = summary.passes_home + summary.passes_away
    completion = (
        summary.pass_completion_home * summary.passes_home
        + summary.pass_completion_away * summary.passes_away
    ) / max(1, passes)
    return {
        "passes": passes,
        "shots": summary.shots_home + summary.shots_away,
        "completion": completion,
        "possession_home": summary.possession_home,
        "xg": summary.micro_xg_home + summary.micro_xg_away,
        "queue": summary.subtick_reception_queue,
    }


def init_worker(root):
    global _WORLD
    from src.simulation.world_cup_runner import build_world_and_tournament

    _WORLD = build_world_and_tournament(root, require_tactics=False)[0]


def run_pair(task):
    home, away, seed, seconds = task
    from src.match_engine.match_micro_runner import run_match_micro_simulation
    from src.memory_engine.poisson_simulator import simulate_match_score

    h = _WORLD.agents[home]
    a = _WORLD.agents[away]
    gh, ga, xh, xa = simulate_match_score(
        h.status_score,
        a.status_score,
        rng=named_rng(seed, "micro_match", home, away, "macro_prior"),
    )
    common = dict(
        home_agent=h,
        away_agent=a,
        goals_home=gh,
        goals_away=ga,
        xg_home=xh,
        xg_away=xa,
        internal_home={},
        internal_away={},
        seed=seed,
        writeback_agents=False,
        match_seconds=seconds,
    )
    start = time.perf_counter()
    baseline = scalar(run_match_micro_simulation(config=cfg(False, seconds), **common))
    split = time.perf_counter()
    treatment = scalar(run_match_micro_simulation(config=cfg(True, seconds), **common))
    end = time.perf_counter()
    delta = {
        key: treatment[key] - baseline[key]
        for key in ("passes", "shots", "completion", "possession_home", "xg")
    }
    return {
        "home": home,
        "away": away,
        "seed": seed,
        "baseline": baseline,
        "treatment": treatment,
        "delta": delta,
        "runtime_s": {"baseline": split - start, "treatment": end - split},
    }


def stats(values):
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.5)),
        "q95": float(np.quantile(values, 0.95)),
    }


def paired_bootstrap(rows, key, repeats=2000, seed=9801):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(sorted((row["home"], row["away"]))), []).append(
            row["delta"][key]
        )
    values = np.asarray([np.mean(value) for value in groups.values()])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "clusters": len(values),
    }


def aggregate(rows, seconds):
    targets = {key: value * seconds / 5400 for key, value in TARGETS.items()}
    errors = {
        key: {
            arm: float(np.mean([abs(row[arm][key] - target) for row in rows]))
            for arm in ("baseline", "treatment")
        }
        for key, target in targets.items()
    }
    effects = {
        key: paired_bootstrap(rows, key)
        for key in ("passes", "shots", "completion", "possession_home", "xg")
    }
    scheduled = sum(row["treatment"]["queue"]["scheduled"] for row in rows)
    applied = sum(row["treatment"]["queue"]["applied"] for row in rows)
    cancelled = sum(row["treatment"]["queue"]["cancelled"] for row in rows)
    runtime = {
        arm: stats([row["runtime_s"][arm] for row in rows])
        for arm in ("baseline", "treatment")
    }
    runtime["treatment_overhead_ratio"] = runtime["treatment"]["mean"] / max(
        runtime["baseline"]["mean"], 1e-9
    )
    return {
        "targets": targets,
        "mean_absolute_target_errors": errors,
        "paired_effects": effects,
        "scheduled": scheduled,
        "applied": applied,
        "cancelled": cancelled,
        "application_rate": applied / max(1, scheduled),
        "runtime": runtime,
    }


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5400.0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=20260727)
    ap.add_argument(
        "--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2))
    )
    ap.add_argument("--teams", nargs="*", default=list(TEAMS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--version", default="9.9.0-shadow")
    ap.add_argument(
        "--output", type=Path, default=ROOT / "reports/acceptance/large_shadow_v98.json"
    )
    return ap


def gates_for(rows, agg, fixture_count, seeds, team_count):
    return {
        "sample_size": len(rows) >= fixture_count * seeds and len(rows) >= 100,
        "team_coverage": team_count >= 12,
        "seed_coverage": seeds >= 8,
        "zero_rng_draws": all(
            row["treatment"]["queue"].get("rng_draws") == 0 for row in rows
        ),
        "queue_exercised": agg["scheduled"] >= 1000,
        "reconciliation_rate": agg["application_rate"] >= 0.95
        and agg["cancelled"] / max(1, agg["scheduled"]) <= 0.05,
        "pass_noninferior": agg["mean_absolute_target_errors"]["passes"]["treatment"]
        <= agg["mean_absolute_target_errors"]["passes"]["baseline"] + 5,
        "shot_noninferior": agg["mean_absolute_target_errors"]["shots"]["treatment"]
        <= agg["mean_absolute_target_errors"]["shots"]["baseline"] + 1,
        "xg_noninferior": agg["mean_absolute_target_errors"]["xg"]["treatment"]
        <= agg["mean_absolute_target_errors"]["xg"]["baseline"] + 0.08,
        "completion_ci_stable": max(
            abs(x) for x in agg["paired_effects"]["completion"]["ci95"]
        )
        <= 0.05,
        "possession_ci_stable": max(
            abs(x) for x in agg["paired_effects"]["possession_home"]["ci95"]
        )
        <= 0.05,
    }


def main():
    args = parser().parse_args()
    fx = fixtures(args.teams)
    tasks = [
        (
            h,
            a,
            derive_seed(args.seed_start, "strict-mirror-shadow", *sorted((h, a)), i),
            args.seconds,
        )
        for h, a in fx
        for i in range(args.seeds)
    ]
    checkpoint = args.output.with_suffix(".checkpoint.json")
    rows = []
    checkpoint_protocol = {
        "seconds": args.seconds,
        "seeds": args.seeds,
        "teams": args.teams,
        "seed_policy": "unordered_fixture_v1",
    }
    if args.resume and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text())
        old = saved.get("protocol", {})
        compatible = old == checkpoint_protocol
        if not compatible:
            raise ValueError("checkpoint protocol does not match requested shadow")
        rows = saved["rows"]
    done = {(r["home"], r["away"], r["seed"]) for r in rows}
    pending = [task for task in tasks if task[:3] not in done]
    start = time.perf_counter()
    if args.workers == 1:
        init_worker(str(ROOT))
        results = (run_pair(task) for task in pending)
        ordered = enumerate(results, 1)
    else:
        pool = ProcessPoolExecutor(
            max_workers=args.workers, initializer=init_worker, initargs=(str(ROOT),)
        )
        future_map = {pool.submit(run_pair, task): task for task in pending}
        ordered = enumerate((future.result() for future in as_completed(future_map)), 1)
    for n, result in ordered:
        rows.append(result)
        rows.sort(key=lambda r: (r["home"], r["away"], r["seed"]))
        if n % max(1, args.workers) == 0 or n == len(pending):
            write_json_atomic(
                checkpoint,
                {"schema_version": 1, "protocol": checkpoint_protocol, "rows": rows},
            )
    if args.workers > 1:
        pool.shutdown()
    agg = aggregate(rows, args.seconds)
    gates = {
        key: bool(value)
        for key, value in gates_for(
            rows, agg, len(fx), args.seeds, len(args.teams)
        ).items()
    }
    elapsed = time.perf_counter() - start
    performance = {
        "wall_s": elapsed,
        "pairs_per_hour": len(pending) / max(elapsed / 3600, 1e-9),
        "resumed_pairs": len(rows) - len(pending),
    }
    protocol = checkpoint_protocol | {
        "dt": 5.5,
        "fixtures": len(fx),
        "seeds_per_fixture": args.seeds,
        "matched_pairs": len(rows),
        "workers": args.workers,
        "bootstrap": "unordered-fixture cluster, 2000 repeats",
    }
    report = {
        "version": args.version,
        "protocol": protocol,
        "aggregate": agg,
        "rows": rows,
        "performance": performance,
        "gates": gates,
        "ready": all(gates.values()),
        "deployment_unchanged": True,
    }
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "pairs": len(rows),
                "performance": performance,
                "paired_effects": agg["paired_effects"],
                "gates": gates,
                "ready": report["ready"],
            },
            indent=2,
        )
    )
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
