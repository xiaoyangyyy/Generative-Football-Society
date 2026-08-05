#!/usr/bin/env python3
"""Validate formal ablations, estimate paired uncertainty, and freeze policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.match_engine.calibration.ablation import ADDITIVE_LAYERS, SUBTRACTIVE_ABLATIONS
from src.match_engine.calibration.contract import evaluate_rows, load_contract, load_statsbomb_baselines
from src.match_engine.calibration.objective import calibration_loss

EXPECTED = tuple([*SUBTRACTIVE_ABLATIONS, *ADDITIVE_LAYERS])
METRICS = (
    "pass_completion", "interceptions_per_pass", "passes_per_team_match",
    "long_pass_share", "shots_per_team_match", "shots_on_target_rate",
    "goals_per_team_match", "fouls_committed_per_team_match",
    "yellow_cards_per_team_match", "red_cards_per_team_match",
    "possession_share", "crosses_per_team_match", "headers_per_team_match",
    "tackles_per_team_match", "micro_xg_per_team_match",
)


def _ordered(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row["fixture"])].append(row)
    return [row for fixture in sorted(buckets) for row in buckets[fixture]]


def _validate(report: dict, variant: str) -> None:
    if report.get("variant") != variant:
        raise ValueError(f"{variant}: report variant mismatch")
    if report.get("protocol") != "six_fixtures_three_seeds_full_90min_matched_to_m0":
        raise ValueError(f"{variant}: wrong protocol")
    rows = report.get("raw_rows", [])
    if report.get("samples_total") != 18 or len(rows) != 18:
        raise ValueError(f"{variant}: expected 18 rows")
    counts = Counter(str(row.get("fixture")) for row in rows)
    if len(counts) != 6 or set(counts.values()) != {3}:
        raise ValueError(f"{variant}: expected six fixtures with three seeds each")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def paired_intervals(
    baseline_rows: list[dict], candidate_rows: list[dict], *, draws: int = 2000, seed: int = 42,
) -> dict:
    base = _ordered(baseline_rows)
    cand = _ordered(candidate_rows)
    if [r["fixture"] for r in base] != [r["fixture"] for r in cand]:
        raise ValueError("candidate fixtures do not match frozen M0")
    rng = random.Random(seed)
    metric_draws = {key: [] for key in METRICS}
    loss_draws: list[float] = []
    contract = load_contract()
    baselines = load_statsbomb_baselines()
    for _ in range(draws):
        idx = [rng.randrange(len(base)) for _ in base]
        b_rows = [base[i] for i in idx]
        c_rows = [cand[i] for i in idx]
        for key in METRICS:
            metric_draws[key].append(mean(float(r[key]) for r in c_rows) - mean(float(r[key]) for r in b_rows))
        loss_draws.append(
            calibration_loss(evaluate_rows(c_rows, contract=contract, baselines=baselines))
            - calibration_loss(evaluate_rows(b_rows, contract=contract, baselines=baselines))
        )
    return {
        "method": "paired_match_bootstrap",
        "draws": draws,
        "seed": seed,
        "delta_loss": {
            "mean": mean(loss_draws),
            "ci95_low": _percentile(loss_draws, 0.025),
            "ci95_high": _percentile(loss_draws, 0.975),
        },
        "metric_mean_deltas": {
            key: {
                "mean": mean(values),
                "ci95_low": _percentile(values, 0.025),
                "ci95_high": _percentile(values, 0.975),
            }
            for key, values in metric_draws.items()
        },
    }


def _decision(variant: str, report: dict, interval: dict) -> tuple[str, str]:
    ci = interval["delta_loss"]
    if variant in ADDITIVE_LAYERS:
        if not report["all_pass"]:
            return "research_only_default_off", "additive layer breaks at least one formal external-validity gate"
        if ci["ci95_high"] < 0:
            return "promotion_candidate_pending_mechanism_validation", "formal external loss improves with a paired 95% interval below zero"
        return "research_only_default_off", "no statistically resolved formal external-loss improvement"
    if not report["all_pass"]:
        return "retain_core", "removal breaks at least one formal external-validity gate"
    if ci["ci95_high"] < 0:
        return "remove_from_default", "removal improves formal external loss with a paired 95% interval below zero"
    if ci["ci95_low"] > 0:
        return "retain_core", "removal worsens formal external loss with a paired 95% interval above zero"
    return "optional_default_off", "removal passes all gates and the paired loss difference is unresolved"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/evaluation/formal_ablation")
    parser.add_argument("--baseline", default="data/evaluation/m0_formal_baseline_v1.json")
    parser.add_argument("--summary", default="data/evaluation/formal_ablation_summary_v1.json")
    parser.add_argument("--policy", default="data/evaluation/module_policy_v1.json")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    baseline = json.loads((ROOT / args.baseline).read_text(encoding="utf-8"))
    baseline_rows = baseline["raw_rows"]
    results = {}
    policy_modules = {
        "shot_model_v7": {"status": "retain_validated", "reason": "held-out non-leaking Brier gate remains authoritative"},
        "narrative_memory_meta_learning": {"status": "research_only_no_real_world_claim", "reason": "external longitudinal causal evidence remains absent"},
    }
    names = {
        "no_affective": "affective_coupling",
        "no_spatial": "spatial_intelligence",
        "no_blend": "statsbomb_pass_logit_blend",
        "no_discipline_tick": "tick_discipline",
        "no_tactical_bias": "tactical_bias",
        "M1": "advanced_world_model",
        "C1": "llm_cognition",
    }
    for variant in EXPECTED:
        path = ROOT / args.input_dir / f"{variant}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate(report, variant)
        interval = paired_intervals(baseline_rows, report["raw_rows"], draws=args.draws)
        status, reason = _decision(variant, report, interval)
        results[variant] = {
            "samples_total": report["samples_total"], "all_pass": report["all_pass"],
            "failed_metrics": report["failed_metrics"], "failed_soft": report["failed_soft"],
            "loss": report["loss"], "baseline_loss": report["baseline_loss"],
            "observed_delta_loss": report["delta_loss"], "paired_uncertainty": interval,
            "decision": status, "decision_reason": reason,
        }
        policy_modules[names[variant]] = {"status": status, "reason": reason, "evidence": str(path.relative_to(ROOT))}
    summary = {
        "schema_version": 1,
        "protocol": "seven_variants_six_fixtures_three_matched_seeds_full_90min",
        "baseline": args.baseline,
        "complete": set(results) == set(EXPECTED),
        "variants": results,
    }
    policy = {
        "schema_version": 2,
        "policy_id": "formal-evidence-first-module-policy-v2",
        "stable_runtime": {"release": "7.0.0", "world_model_default": "off", "llm_cognition_default": "off"},
        "formal_evidence": args.summary,
        "modules": policy_modules,
    }
    for rel, payload in ((args.summary, summary), (args.policy, policy)):
        out = ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": summary["complete"], "variants": {k: v["decision"] for k, v in results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
