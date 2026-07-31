"""Persistent outcome-linked audits for world-model/LLM tactical fusion."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.simulation.counterfactual import evaluate_intervention


FUSION_AUDIT_SCHEMA_VERSION = 1
DEFAULT_FUSION_AUDIT_PATH = Path("data/persistence/world_model_fusion.jsonl")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def fusion_audit_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / DEFAULT_FUSION_AUDIT_PATH


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _finite(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def build_outcome_linked_fusion_record(
    *,
    decision_id: str,
    team: str,
    opponent: str,
    stage: str,
    decision_audit: dict[str, Any] | None,
    outcomes: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact JSON-safe record without copying the full model packet."""
    audit = _json_safe(dict(decision_audit or {}))
    result = str(outcomes.get("result", "draw"))
    if result not in {"win", "draw", "loss"}:
        result = "draw"
    return {
        "schema_version": FUSION_AUDIT_SCHEMA_VERSION,
        "decision_id": str(decision_id),
        "team": str(team),
        "opponent": str(opponent),
        "stage": str(stage),
        "decision": audit,
        "outcome": {
            "result": result,
            "points": {"win": 3.0, "draw": 1.0, "loss": 0.0}[result],
            "goal_diff": _finite(outcomes.get("goal_diff")),
            "xg_diff": _finite(outcomes.get("xg_for"))
            - _finite(outcomes.get("xg_against")),
            "fatigue_delta": _finite(outcomes.get("fatigue_delta")),
        },
    }


def append_fusion_audits(
    base_dir: str | Path,
    records: Iterable[dict[str, Any]],
) -> Path:
    """Append records durably; loaders de-duplicate deterministic decision ids."""
    path = fusion_audit_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def load_fusion_audits(path: str | Path) -> list[dict[str, Any]]:
    """Load valid records and retain the last copy of duplicate decision ids."""
    source = Path(path)
    if not source.is_file():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid fusion audit JSON at line {line_number}"
                ) from exc
            if int(record.get("schema_version", 0)) != FUSION_AUDIT_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported fusion audit schema at line {line_number}"
                )
            decision_id = str(record.get("decision_id", ""))
            if not decision_id:
                raise ValueError(
                    f"missing fusion decision_id at line {line_number}"
                )
            by_id[decision_id] = record
    return list(by_id.values())


def _mean(records: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def _outcome_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(records),
        "mean_points": _mean(records, ("outcome", "points")),
        "mean_goal_diff": _mean(records, ("outcome", "goal_diff")),
        "mean_xg_diff": _mean(records, ("outcome", "xg_diff")),
        "mean_fatigue_delta": _mean(records, ("outcome", "fatigue_delta")),
    }


def evaluate_fusion_audits(
    records: Iterable[dict[str, Any]],
    *,
    min_records: int = 20,
) -> dict[str, Any]:
    """Evaluate coverage and observational outcomes without causal overclaiming."""
    rows = list(records)
    evidence = [
        row for row in rows
        if bool((row.get("decision") or {}).get("evidence_available"))
    ]
    agreements = [
        row for row in evidence
        if not bool(row["decision"].get("disagreed_with_recommendation"))
    ]
    disagreements = [
        row for row in evidence
        if bool(row["decision"].get("disagreed_with_recommendation"))
    ]
    constrained = [
        row for row in evidence
        if bool(row["decision"].get("selection_constrained"))
    ]
    rationale = [
        row for row in evidence
        if str(row["decision"].get("rationale", "")).strip()
    ]
    recommendations = Counter(
        str(row["decision"].get("recommended_tactical_preset", "none"))
        for row in evidence
    )
    selection_advantages = []
    for row in evidence:
        decision = row["decision"]
        selected = decision.get("selected_evidence") or {}
        balanced = decision.get("balanced_evidence") or {}
        if "risk_adjusted_value" in selected and "risk_adjusted_value" in balanced:
            selection_advantages.append(
                _finite(selected["risk_adjusted_value"])
                - _finite(balanced["risk_adjusted_value"])
            )
    evidence_count = len(evidence)
    gates = {
        "minimum_records": len(rows) >= int(min_records),
        "has_open_quality_gates": evidence_count > 0,
        "complete_rationales": len(rationale) == evidence_count,
        "no_forced_out_of_set_choices": len(constrained) == 0,
    }
    return {
        "schema_version": FUSION_AUDIT_SCHEMA_VERSION,
        "evaluation_kind": "observational_world_model_llm_fusion",
        "records": len(rows),
        "evidence_records": evidence_count,
        "gate_open_rate": evidence_count / max(1, len(rows)),
        "agreement_rate": len(agreements) / max(1, evidence_count),
        "constraint_rate": len(constrained) / max(1, evidence_count),
        "rationale_coverage": len(rationale) / max(1, evidence_count),
        "mean_observation_coverage": _mean(
            evidence, ("decision", "observation_coverage"),
        ),
        "mean_recommendation_confidence": _mean(
            evidence, ("decision", "recommendation_confidence"),
        ),
        "mean_recommendation_margin": _mean(
            evidence, ("decision", "recommendation_margin"),
        ),
        "mean_selected_advantage_vs_balanced": (
            float(np.mean(selection_advantages)) if selection_advantages else 0.0
        ),
        "recommendation_distribution": dict(sorted(recommendations.items())),
        "observed_outcomes": {
            "all": _outcome_summary(rows),
            "agreement": _outcome_summary(agreements),
            "disagreement": _outcome_summary(disagreements),
        },
        "gates": gates,
        "ready_for_observational_comparison": all(gates.values()),
        "causal_interpretation": False,
        "limitations": [
            "Agreement groups are self-selected, not randomized treatments.",
            "Model value is a short-horizon proxy, not match win probability.",
            "Use matched-seed simulation experiments before claiming uplift.",
        ],
    }


def evaluate_matched_seed_tactical_policy(
    run,
    *,
    baseline_preset: str,
    treatment_preset: str,
    root_seed: int = 42,
    samples: int = 16,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Estimate a tactical effect inside the simulator with paired random seeds."""
    report = evaluate_intervention(
        run,
        baseline_preset,
        treatment_preset,
        root_seed=root_seed,
        samples=samples,
        intervention=f"{baseline_preset}_to_{treatment_preset}",
        confidence=confidence,
    )
    return {
        "evaluation_kind": "matched_seed_simulator_counterfactual",
        "baseline_preset": baseline_preset,
        "treatment_preset": treatment_preset,
        "samples": report.samples,
        "baseline_mean_utility": report.baseline_mean,
        "treatment_mean_utility": report.treatment_mean,
        "average_treatment_effect": report.average_treatment_effect,
        "standard_error": report.standard_error,
        "confidence_interval": [report.ci_low, report.ci_high],
        "directionally_supported": report.directionally_supported,
        "paired_deltas": list(report.paired_deltas),
        "causal_interpretation": "within_simulator_only",
        "limitations": [
            "The estimate is causal only for the configured simulator.",
            "External football validity requires real-world evaluation.",
            "The opponent policy is held fixed across paired runs.",
        ],
    }
