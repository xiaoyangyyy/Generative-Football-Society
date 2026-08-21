"""Replayable world-model advice and explicit manager-adoption evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping

from src.product.match_plan import PLAYABLE_TACTICS


ADVICE_SCHEMA_VERSION = 1
ADOPTION_SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 1
ADVICE_INTENTS = {"adopt_recommendation", "reviewed_then_selected"}
_EVENTS = ("retain", "turnover", "shot", "foul", "out")
_BOUNDARY = (
    "local calibrated-candidate short-horizon policy proxy for this deterministic "
    "simulated pre-match state; advisory only, not a score or win-probability "
    "forecast, causal estimate, real-football recommendation or medical judgment"
)


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _number(value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("world-model advice metric is invalid")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError("world-model advice metric is outside its bounded domain")
    return round(number, 9)


def _counter(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000_000:
        raise ValueError("world-model advice evidence count is invalid")
    return value


def _candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("world-model tactical candidate is invalid")
    tactic = str(row.get("tactical_preset") or "")
    if tactic not in PLAYABLE_TACTICS:
        raise ValueError("world-model advice contains an unsupported tactic")
    events = row.get("event_probabilities")
    if not isinstance(events, Mapping) or set(events) != set(_EVENTS):
        raise ValueError("world-model advice event distribution is invalid")
    probabilities = {
        name: _number(events[name], minimum=0.0, maximum=1.0)
        for name in _EVENTS
    }
    if abs(sum(probabilities.values()) - 1.0) > 1e-6:
        raise ValueError("world-model advice event probabilities do not sum to one")
    return {
        "tactic": tactic,
        "risk_adjusted_value": _number(
            row.get("risk_adjusted_value"), minimum=-10.0, maximum=10.0,
        ),
        "effective_confidence": _number(
            row.get("effective_confidence"), minimum=0.0, maximum=1.0,
        ),
        "uncertainty": _number(
            row.get("uncertainty"), minimum=0.0, maximum=1.0,
        ),
        "fatigue_cost_proxy": _number(
            row.get("fatigue_cost_proxy"), minimum=0.0, maximum=1.0,
        ),
        "structural_risk_proxy": _number(
            row.get("structural_risk_proxy"), minimum=0.0, maximum=1.0,
        ),
        "event_probabilities": probabilities,
    }


def _state_sources(
    snapshots: Mapping[str, Mapping[str, Any]], *, home: str, away: str,
) -> dict[str, dict[str, str]]:
    if not isinstance(snapshots, Mapping) or set(snapshots) != {home, away}:
        raise ValueError("world-model advice state coverage mismatch")
    result = {}
    for team in (home, away):
        row = snapshots[team]
        source = str(row.get("source") or "") if isinstance(row, Mapping) else ""
        identity = (
            str(row.get("source_identity") or "")
            if isinstance(row, Mapping) else ""
        )
        if (
            source not in {
                "deterministic_team_baseline", "deterministic_roster_baseline",
                "persisted_carryover",
            }
            or not re.fullmatch(r"[0-9a-f]{64}", identity)
        ):
            raise ValueError("world-model advice state identity is invalid")
        result[team] = {
            "team_id": team, "source": source, "source_identity": identity,
        }
    return result


def build_manager_decision_advice(
    *, packet: Mapping[str, Any], season_id: str, fixture_id: str,
    home: str, away: str, manager_team: str, mode: str,
    match_seed: int, base_revision: int,
    checkpoint_artifact: str, checkpoint_sha256: str,
    state_snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compact the engine packet into a bounded, persistable product contract."""
    if mode not in {"research", "cognitive"}:
        raise ValueError("world-model manager advice requires research mode")
    if manager_team not in {home, away}:
        raise ValueError("world-model advice manager-team identity mismatch")
    if (
        not isinstance(match_seed, int) or isinstance(match_seed, bool)
        or not 0 <= match_seed <= 2**31 - 1
        or not isinstance(base_revision, int) or isinstance(base_revision, bool)
        or base_revision < 0
    ):
        raise ValueError("world-model advice revision or seed is invalid")
    if not isinstance(packet, Mapping) or packet.get("available") is not True:
        raise ValueError("world-model advice requires an available decision packet")
    if packet.get("evaluation_scope") != "short_horizon_tactical_policy_proxy":
        raise ValueError("world-model advice scope mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_sha256 or "")):
        raise ValueError("world-model advice checkpoint identity is invalid")
    artifact = str(checkpoint_artifact or "")
    if not artifact or len(artifact) > 320:
        raise ValueError("world-model advice checkpoint artifact is invalid")
    runtime_signature = str(packet.get("checkpoint_signature") or "")
    if runtime_signature != f"sha256:{checkpoint_sha256}":
        raise ValueError("world-model advice runtime checkpoint mismatch")

    candidates = [_candidate(row) for row in packet.get("candidates") or []]
    if {row["tactic"] for row in candidates} != set(PLAYABLE_TACTICS) or (
        len(candidates) != len(PLAYABLE_TACTICS)
    ):
        raise ValueError("world-model advice must compare every playable tactic once")
    candidates.sort(key=lambda row: (-row["risk_adjusted_value"], row["tactic"]))
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    recommended = candidates[0]["tactic"]
    margin = round(
        candidates[0]["risk_adjusted_value"]
        - candidates[1]["risk_adjusted_value"],
        9,
    )
    reliability = packet.get("fusion_reliability") or {}
    if not isinstance(reliability, Mapping):
        raise ValueError("world-model advice reliability evidence is invalid")
    state_sources = _state_sources(state_snapshots, home=home, away=away)
    payload = {
        "schema_version": ADVICE_SCHEMA_VERSION,
        "season_id": str(season_id), "fixture_id": str(fixture_id),
        "home": home, "away": away, "manager_team": manager_team,
        "mode": mode, "match_seed": match_seed,
        "base_revision": base_revision, "issued_revision": base_revision + 1,
        "checkpoint": {
            "artifact": artifact,
            "sha256": str(checkpoint_sha256),
            "runtime_signature": runtime_signature,
        },
        "representative_state": {
            "kind": "deterministic_prematch_policy_proxy",
            "carryover_sources": state_sources,
            "identity": _identity({
                "season_id": str(season_id), "fixture_id": str(fixture_id),
                "match_seed": match_seed, "sources": state_sources,
            }),
        },
        "evaluation_scope": "short_horizon_tactical_policy_proxy",
        "horizon_s": _number(packet.get("horizon_s"), minimum=1.0, maximum=120.0),
        "observation_coverage": _number(
            packet.get("observation_coverage"), minimum=0.0, maximum=1.0,
        ),
        "recommended_tactic": recommended,
        "recommendation_margin": margin,
        "recommendation_confidence": _number(
            candidates[0]["effective_confidence"], minimum=0.0, maximum=1.0,
        ),
        "reliability": {
            "evidence_tier": str(
                reliability.get("evidence_tier", "insufficient_history")
            )[:64],
            "guidance": str(reliability.get("guidance", "low_authority"))[:64],
            "adjusted_recommendation_trust": _number(
                reliability.get("adjusted_recommendation_trust", 0.0),
                minimum=0.0, maximum=1.0,
            ),
            "matched_seed_samples": _counter(
                reliability.get("matched_seed_samples", 0)
            ),
            "historical_match_records": _counter(
                reliability.get("historical_match_records", 0)
            ),
            "causal_scope": str(reliability.get("causal_scope", "none"))[:64],
        },
        "candidates": candidates,
        "claim_boundary": _BOUNDARY,
    }
    payload["advice_identity"] = _identity(payload)
    validate_manager_decision_advice(payload)
    return payload


def validate_manager_decision_advice(advice: Mapping[str, Any]) -> None:
    if not isinstance(advice, Mapping):
        raise ValueError("world-model manager advice is invalid")
    frozen = copy.deepcopy(dict(advice))
    identity = frozen.pop("advice_identity", None)
    expected_keys = {
        "schema_version", "season_id", "fixture_id", "home", "away",
        "manager_team", "mode", "match_seed", "base_revision", "issued_revision",
        "checkpoint", "representative_state", "evaluation_scope", "horizon_s",
        "observation_coverage", "recommended_tactic", "recommendation_margin",
        "recommendation_confidence", "reliability", "candidates",
        "claim_boundary", "advice_identity",
    }
    if (
        set(advice) != expected_keys
        or advice.get("schema_version") != ADVICE_SCHEMA_VERSION
        or advice.get("mode") not in {"research", "cognitive"}
        or advice.get("manager_team") not in {advice.get("home"), advice.get("away")}
        or advice.get("evaluation_scope") != "short_horizon_tactical_policy_proxy"
        or advice.get("claim_boundary") != _BOUNDARY
        or identity != _identity(frozen)
    ):
        raise ValueError("world-model manager advice identity mismatch")
    for field, maximum in (
        ("season_id", 128), ("fixture_id", 128),
        ("home", 160), ("away", 160), ("manager_team", 160),
    ):
        value = advice.get(field)
        if not isinstance(value, str) or not value or len(value) > maximum:
            raise ValueError("world-model manager advice subject identity is invalid")
    match_seed = advice.get("match_seed")
    if (
        isinstance(match_seed, bool) or not isinstance(match_seed, int)
        or not 0 <= match_seed <= 2**31 - 1
    ):
        raise ValueError("world-model manager advice seed is invalid")
    base_revision = advice.get("base_revision")
    if (
        isinstance(base_revision, bool) or not isinstance(base_revision, int)
        or base_revision < 0 or advice.get("issued_revision") != base_revision + 1
    ):
        raise ValueError("world-model manager advice revision mismatch")
    checkpoint = advice.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"artifact", "sha256", "runtime_signature"}
        or not str(checkpoint.get("artifact") or "")
        or len(str(checkpoint.get("artifact") or "")) > 320
        or not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256") or ""))
        or checkpoint.get("runtime_signature")
        != f"sha256:{checkpoint.get('sha256')}"
    ):
        raise ValueError("world-model manager advice checkpoint mismatch")
    state = advice.get("representative_state")
    sources = state.get("carryover_sources") if isinstance(state, Mapping) else None
    expected_state_identity = _identity({
        "season_id": advice.get("season_id"),
        "fixture_id": advice.get("fixture_id"),
        "match_seed": advice.get("match_seed"),
        "sources": sources,
    })
    if (
        not isinstance(state, Mapping)
        or set(state) != {"kind", "carryover_sources", "identity"}
        or state.get("kind") != "deterministic_prematch_policy_proxy"
        or state.get("identity") != expected_state_identity
    ):
        raise ValueError("world-model manager advice state mismatch")
    _state_sources(sources, home=str(advice.get("home")), away=str(advice.get("away")))
    _number(advice.get("horizon_s"), minimum=1.0, maximum=120.0)
    _number(advice.get("observation_coverage"), minimum=0.0, maximum=1.0)
    _number(advice.get("recommendation_confidence"), minimum=0.0, maximum=1.0)
    candidates = advice.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(PLAYABLE_TACTICS):
        raise ValueError("world-model manager advice candidates are invalid")
    canonical = []
    for rank, row in enumerate(candidates, start=1):
        if not isinstance(row, Mapping) or set(row) != {
            "tactic", "risk_adjusted_value", "effective_confidence", "uncertainty",
            "fatigue_cost_proxy", "structural_risk_proxy", "event_probabilities",
            "rank",
        } or row.get("rank") != rank:
            raise ValueError("world-model manager advice candidate rank mismatch")
        rebuilt = _candidate({
            "tactical_preset": row.get("tactic"),
            **{key: row.get(key) for key in (
                "risk_adjusted_value", "effective_confidence", "uncertainty",
                "fatigue_cost_proxy", "structural_risk_proxy", "event_probabilities",
            )},
        })
        rebuilt["rank"] = rank
        canonical.append(rebuilt)
    if canonical != candidates or {row["tactic"] for row in canonical} != set(PLAYABLE_TACTICS):
        raise ValueError("world-model manager advice candidate replay mismatch")
    expected_order = sorted(
        canonical, key=lambda row: (-row["risk_adjusted_value"], row["tactic"]),
    )
    expected_margin = round(
        expected_order[0]["risk_adjusted_value"]
        - expected_order[1]["risk_adjusted_value"], 9,
    )
    if (
        canonical != expected_order
        or advice.get("recommended_tactic") != expected_order[0]["tactic"]
        or advice.get("recommendation_margin") != expected_margin
        or advice.get("recommendation_confidence")
        != expected_order[0]["effective_confidence"]
    ):
        raise ValueError("world-model manager advice recommendation replay mismatch")
    reliability = advice.get("reliability")
    if not isinstance(reliability, Mapping) or set(reliability) != {
        "evidence_tier", "guidance", "adjusted_recommendation_trust",
        "matched_seed_samples", "historical_match_records", "causal_scope",
    }:
        raise ValueError("world-model manager advice reliability mismatch")
    if any(len(str(reliability.get(key) or "")) > 64 for key in (
        "evidence_tier", "guidance", "causal_scope",
    )):
        raise ValueError("world-model manager advice reliability label is invalid")
    _number(
        reliability.get("adjusted_recommendation_trust"), minimum=0.0, maximum=1.0,
    )
    _counter(reliability.get("matched_seed_samples"))
    _counter(reliability.get("historical_match_records"))


def _comparison_authority(advice: Mapping[str, Any]) -> dict[str, Any]:
    reliability = advice["reliability"]
    reasons = []
    if advice["recommendation_confidence"] < 0.15:
        reasons.append("low_model_confidence")
    if reliability["adjusted_recommendation_trust"] < 0.25:
        reasons.append("low_historical_trust")
    if advice["recommendation_margin"] < 0.005:
        reasons.append("narrow_top_two_margin")
    if reliability["evidence_tier"] == "insufficient_history":
        reasons.append("insufficient_history")
    if reliability["guidance"] == "low_authority":
        reasons.append("low_authority_guidance")
    return {
        "level": "exploratory_only" if reasons else "bounded_review",
        "reasons": reasons,
        "display_thresholds": {
            "minimum_confidence": 0.15,
            "minimum_historical_trust": 0.25,
            "minimum_top_two_margin": 0.005,
        },
        "automatic_adoption_authorized": False,
        "performance_validated": False,
    }


def build_manager_advice_comparison(
    advice: Mapping[str, Any], *, selected_tactic: str,
) -> dict[str, Any]:
    """Compare the recommendation with the manager's current tactic selection."""
    validate_manager_decision_advice(advice)
    if selected_tactic not in PLAYABLE_TACTICS:
        raise ValueError("world-model advice comparison tactic is invalid")
    recommended = next(
        row for row in advice["candidates"]
        if row["tactic"] == advice["recommended_tactic"]
    )
    selected = next(
        row for row in advice["candidates"]
        if row["tactic"] == selected_tactic
    )
    fields = (
        "risk_adjusted_value", "effective_confidence", "uncertainty",
        "fatigue_cost_proxy", "structural_risk_proxy",
    )
    deltas = {
        field: round(float(recommended[field]) - float(selected[field]), 9)
        for field in fields
    }
    event_deltas = {
        event: round(
            float(recommended["event_probabilities"][event])
            - float(selected["event_probabilities"][event]),
            9,
        )
        for event in _EVENTS
    }
    tradeoffs = []
    for field, lower_is_better in (
        ("uncertainty", True),
        ("fatigue_cost_proxy", True),
        ("structural_risk_proxy", True),
    ):
        delta = deltas[field]
        if delta == 0.0:
            label = "equal"
        elif (delta < 0.0) == lower_is_better:
            label = "recommended_lower"
        else:
            label = "recommended_higher"
        tradeoffs.append({"metric": field, "comparison": label})
    payload = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "advice_identity": advice["advice_identity"],
        "selected_tactic": selected_tactic,
        "recommended_tactic": advice["recommended_tactic"],
        "aligned_with_recommendation": (
            selected_tactic == advice["recommended_tactic"]
        ),
        "selected_rank": selected["rank"],
        "recommended_rank": recommended["rank"],
        "authority": _comparison_authority(advice),
        "recommended_minus_selected": {
            **deltas,
            "event_probabilities": event_deltas,
        },
        "tradeoffs": tradeoffs,
        "claim_boundary": (
            "identity-bound comparison of two short-horizon simulator policy "
            "proxies; display thresholds disclose evidence weakness only and do "
            "not estimate score, win probability, causal effect or real-football "
            "performance"
        ),
    }
    payload["comparison_identity"] = _identity(payload)
    return payload


def validate_manager_advice_comparison(
    comparison: Mapping[str, Any], *, advice: Mapping[str, Any],
    selected_tactic: str,
) -> None:
    if not isinstance(comparison, Mapping):
        raise ValueError("world-model advice comparison is invalid")
    expected = build_manager_advice_comparison(
        advice, selected_tactic=selected_tactic,
    )
    if dict(comparison) != expected:
        raise ValueError("world-model advice comparison replay mismatch")


def build_manager_advice_adoption(
    advice: Mapping[str, Any], *, selected_tactic: str, intent: str,
) -> dict[str, Any]:
    """Bind an explicit UI intent and final selected tactic to one advice packet."""
    validate_manager_decision_advice(advice)
    if selected_tactic not in PLAYABLE_TACTICS or intent not in ADVICE_INTENTS:
        raise ValueError("world-model advice adoption choice is invalid")
    recommended = str(advice["recommended_tactic"])
    if intent == "adopt_recommendation" and selected_tactic != recommended:
        raise ValueError("adopted world-model recommendation does not match tactic")
    selected = next(
        row for row in advice["candidates"] if row["tactic"] == selected_tactic
    )
    payload = {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "advice_identity": advice["advice_identity"],
        "intent": intent,
        "selected_tactic": selected_tactic,
        "recommended_tactic": recommended,
        "aligned_with_recommendation": selected_tactic == recommended,
        "selected_rank": selected["rank"],
        "selected_risk_adjusted_value": selected["risk_adjusted_value"],
        "claim_boundary": (
            "records the manager's explicit review/adoption interaction and final "
            "simulated tactic only; it does not prove decision quality or causality"
        ),
    }
    payload["adoption_identity"] = _identity(payload)
    return payload


def validate_manager_advice_adoption(
    adoption: Mapping[str, Any], *, advice: Mapping[str, Any],
    selected_tactic: str,
) -> None:
    if not isinstance(adoption, Mapping):
        raise ValueError("world-model advice adoption is invalid")
    expected = build_manager_advice_adoption(
        advice, selected_tactic=selected_tactic,
        intent=str(adoption.get("intent") or ""),
    )
    if dict(adoption) != expected:
        raise ValueError("world-model advice adoption replay mismatch")


__all__ = [
    "ADVICE_INTENTS", "build_manager_advice_adoption",
    "build_manager_advice_comparison", "build_manager_decision_advice",
    "validate_manager_advice_adoption", "validate_manager_advice_comparison",
    "validate_manager_decision_advice",
]
