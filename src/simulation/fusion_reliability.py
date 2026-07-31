"""Conservative historical trust for world-model/LLM tactical fusion."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def summarize_team_strategy_memory(
    audit_records: Iterable[dict[str, Any]],
    *,
    team: str,
    opponent: str = "",
    recent_limit: int = 5,
) -> dict[str, Any]:
    """Compress persistent tactical history without treating it as causal proof."""
    rows = [row for row in audit_records if str(row.get("team", "")) == team]
    by_preset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        decision = row.get("decision") or {}
        preset = str(decision.get("selected_tactical_preset", "unknown"))
        by_preset[preset].append(row)
    preset_summaries = {}
    for preset, group in sorted(by_preset.items()):
        outcomes = [row.get("outcome") or {} for row in group]
        preset_summaries[preset] = {
            "samples": len(group),
            "mean_points": _mean([_finite(item.get("points")) for item in outcomes]),
            "mean_goal_diff": _mean([
                _finite(item.get("goal_diff")) for item in outcomes
            ]),
            "mean_xg_diff": _mean([
                _finite(item.get("xg_diff")) for item in outcomes
            ]),
            "mean_fatigue_delta": _mean([
                _finite(item.get("fatigue_delta")) for item in outcomes
            ]),
        }
    opponent_rows = [
        row for row in rows if opponent and str(row.get("opponent", "")) == opponent
    ]
    recent = []
    for row in (opponent_rows or rows)[-max(1, int(recent_limit)):]:
        decision = row.get("decision") or {}
        outcome = row.get("outcome") or {}
        recent.append({
            "opponent": str(row.get("opponent", "")),
            "stage": str(row.get("stage", "")),
            "selected_tactical_preset": str(
                decision.get("selected_tactical_preset", "unknown")
            ),
            "world_model_evidence_available": bool(
                decision.get("evidence_available", False)
            ),
            "followed_world_model": (
                not bool(decision.get("disagreed_with_recommendation", False))
                if bool(decision.get("evidence_available", False)) else None
            ),
            "result": str(outcome.get("result", "draw")),
            "goal_diff": _finite(outcome.get("goal_diff")),
            "xg_diff": _finite(outcome.get("xg_diff")),
        })
    return {
        "team": team,
        "opponent": opponent,
        "matches": len(rows),
        "opponent_specific_matches": len(opponent_rows),
        "by_selected_tactical_preset": preset_summaries,
        "recent_relevant_decisions": recent,
        "causal_interpretation": False,
        "policy": (
            "Use as weak historical context; do not infer that a tactic caused "
            "the observed result."
        ),
    }


def build_fusion_reliability_profile(
    *,
    team: str,
    recommended_preset: str,
    raw_recommendation_confidence: float,
    audit_records: Iterable[dict[str, Any]],
    counterfactual_records: Iterable[dict[str, Any]],
    quality_gate_open: bool,
) -> dict[str, Any]:
    """Choose an evidence tier; history can reduce trust but never open a gate."""
    team_audits = [
        row for row in audit_records
        if str(row.get("team", "")) == team
        and bool((row.get("decision") or {}).get("evidence_available", False))
    ]
    relevant_cf = [
        row for row in counterfactual_records
        if str(row.get("team", "")) == team
        and str(row.get("treatment_preset", "")) == recommended_preset
        and str(row.get("baseline_preset", "balanced")) == "balanced"
        and str(row.get("causal_interpretation", "")) == "within_simulator_only"
        and not bool(row.get("fast_mode", False))
        and _finite(row.get("match_seconds")) >= 600.0
    ]
    matched_samples = sum(int(row.get("samples", 0)) for row in relevant_cf)
    positive_samples = 0
    negative_samples = 0
    effects = []
    for row in relevant_cf:
        samples = int(row.get("samples", 0))
        interval = row.get("confidence_interval") or [0.0, 0.0]
        low = _finite(interval[0]) if len(interval) else 0.0
        high = _finite(interval[1]) if len(interval) > 1 else 0.0
        effect = _finite(row.get("average_treatment_effect"))
        effects.extend([effect] * max(0, samples))
        if low > 0.0:
            positive_samples += samples
        elif high < 0.0:
            negative_samples += samples

    if matched_samples >= 8:
        tier = "matched_seed_simulator"
        if positive_samples > negative_samples:
            multiplier = 1.10
            guidance = "moderate_support"
        elif negative_samples > positive_samples:
            multiplier = 0.70
            guidance = "reduce_reliance"
        else:
            multiplier = 0.90
            guidance = "inconclusive"
    elif len(team_audits) >= 5:
        tier = "observational_only"
        multiplier = 0.95
        guidance = "advisory_only"
    else:
        tier = "insufficient_history"
        multiplier = 0.85
        guidance = "low_authority"

    raw_confidence = float(np.clip(
        _finite(raw_recommendation_confidence), 0.0, 1.0,
    ))
    adjusted = raw_confidence * multiplier if quality_gate_open else 0.0
    return {
        "evidence_tier": tier,
        "guidance": guidance,
        "historical_match_records": len(team_audits),
        "matched_seed_samples": matched_samples,
        "matched_seed_mean_effect": _mean(effects),
        "positive_directional_samples": positive_samples,
        "negative_directional_samples": negative_samples,
        "raw_recommendation_confidence": raw_confidence,
        "trust_multiplier": multiplier,
        "adjusted_recommendation_trust": float(np.clip(adjusted, 0.0, 1.0)),
        "quality_gate_open": bool(quality_gate_open),
        "can_open_quality_gate": False,
        "causal_scope": (
            "within_simulator_only" if tier == "matched_seed_simulator"
            else "none"
        ),
        "policy": (
            "Historical evidence may lower advisory trust but can never reopen "
            "a closed world-model quality gate."
        ),
    }


def enrich_decision_packet_with_history(
    packet: dict[str, Any],
    *,
    team: str,
    opponent: str,
    audit_records: Iterable[dict[str, Any]],
    counterfactual_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(packet)
    audits = list(audit_records)
    counterfactuals = list(counterfactual_records)
    recommended = str(enriched.get("recommended_tactical_preset", "none"))
    enriched["strategy_memory"] = summarize_team_strategy_memory(
        audits, team=team, opponent=opponent,
    )
    enriched["fusion_reliability"] = build_fusion_reliability_profile(
        team=team,
        recommended_preset=recommended,
        raw_recommendation_confidence=_finite(
            enriched.get("recommendation_confidence")
        ),
        audit_records=audits,
        counterfactual_records=counterfactuals,
        quality_gate_open=bool(enriched.get("available")),
    )
    return enriched
