from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from src.match_engine.cognitive.config import CognitiveMatchConfig
from src.match_engine.cognitive.events import (
    ENTITY_TIER_COACH,
    CognitiveTriggerEvent,
)
from src.match_engine.cognitive.executor import CognitiveExecutor
from src.match_engine.state import (
    CoachAffectiveState,
    CrowdState,
    MatchAffectiveState,
    RefereeAffectiveState,
    TeamAffectiveState,
)
from src.simulation.agent import SocietyAgent
from src.simulation.cross_match_state import (
    TeamSquadCarryover,
    apply_carryover_to_agent,
)
from src.simulation.meta_learning import MetaLearningController
from src.simulation.match_pipeline import finalize_match_feedback
from src.simulation.narrative_events import NarrativeEventBus
from src.simulation.psychological_state import PsychologicalState
from src.simulation.society_continuity import (
    build_society_decision_context,
    capture_society_continuity,
    society_public_snapshot,
    society_public_transition,
    validate_society_public_snapshot,
    validate_society_continuity_state,
)


def _agent(name: str, seed: int = 71) -> SocietyAgent:
    return SocietyAgent(
        name,
        {
            "tier": "Semi-Core",
            "final_status_score": 57.0,
            "c1_win_rate": 51.0,
            "c3_major_exp": 43.0,
            "c5_pressure": 48.0,
        },
        random_root_seed=seed,
    )


def _rehash(state: dict) -> None:
    payload = copy.deepcopy(state)
    payload.pop("state_identity", None)
    state["state_identity"] = hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _match_state() -> MatchAffectiveState:
    return MatchAffectiveState(
        home=TeamAffectiveState(
            team_id="Alpha",
            coach=CoachAffectiveState(
                team_id="Alpha",
                tactical_current={
                    "pressing_intensity": 0.5,
                    "risk_budget": 0.5,
                },
            ),
        ),
        away=TeamAffectiveState(
            team_id="Beta",
            coach=CoachAffectiveState(team_id="Beta"),
        ),
        referee=RefereeAffectiveState(strictness_base=0.55),
        crowd=CrowdState(),
    )


def test_society_state_round_trip_restores_memory_psychology_and_meta_evidence():
    source = _agent("Alpha")
    memory = source._register_memory_event(
        "A bounded lesson from the previous match",
        event_type="decision_event",
        source="match_log",
    )
    source.psychological_state = PsychologicalState(
        morale=-0.2,
        pressure=0.6,
        trust=0.3,
        risk_appetite=-0.4,
        conflict=0.2,
        audience_activation=0.1,
    )
    audit = source.apply_llm_reflection({
        "reflection": "Reduce risk after the previous evidence.",
        "confidence": 1.0,
        "evidence_memory_ids": [memory["id"]],
        "suggested_adjustments": {"risk_budget": -0.08},
    }, operation_id="reflection-md01")
    state = capture_society_continuity(
        source, source_transaction_id="season-1:fixture-1",
    )
    carry = TeamSquadCarryover(team_id="Alpha", society_state=state)
    restored_carry = TeamSquadCarryover.from_dict(json.loads(json.dumps(
        carry.to_dict(), ensure_ascii=False, allow_nan=False,
    )))
    restored = _agent("Alpha")
    restored.squad_carryover = restored_carry

    apply_carryover_to_agent(restored)

    assert restored.psychological_state == source.psychological_state
    assert restored.tactical_controls == source.tactical_controls
    assert restored.episodic_memory == source.episodic_memory
    assert restored.memory_event_log == source.memory_event_log
    assert restored.llm_reflection_audit[-1] == audit
    proposal = MetaLearningController().propose(restored, {
        "confidence": 1.0,
        "evidence_memory_ids": [memory["id"]],
        "suggested_adjustments": {"pressing_intensity": 0.04},
    })
    assert proposal.evidence_ids == (memory["id"],)
    context = build_society_decision_context(restored)
    assert context is not None
    assert context["source_state_identity"] == state["state_identity"]
    assert context["high_salience_memories"][0]["id"] == memory["id"]


def test_matched_seed_targeted_narrative_changes_only_targeted_agent_state():
    control_alpha = _agent("Alpha", 91)
    control_beta = _agent("Beta", 91)
    treated_alpha = _agent("Alpha", 91)
    treated_beta = _agent("Beta", 91)
    NarrativeEventBus().publish({
        "key_event": "Supporters question the late tactical retreat.",
        "signals": {
            "coach_pressure": 0.8,
            "crowd_hostility": 0.7,
            "player_anxiety": 0.5,
            "media_amplification": 0.9,
            "unity_signal": -0.2,
        },
        "targets": ["Alpha"],
        "persistence": 0.8,
    }, [treated_alpha, treated_beta], stage="md-01")

    control_target = capture_society_continuity(
        control_alpha, source_transaction_id="matched-seed-md01",
    )
    treated_target = capture_society_continuity(
        treated_alpha, source_transaction_id="matched-seed-md01",
    )
    control_other = capture_society_continuity(
        control_beta, source_transaction_id="matched-seed-md01",
    )
    treated_other = capture_society_continuity(
        treated_beta, source_transaction_id="matched-seed-md01",
    )

    assert treated_target != control_target
    assert treated_target["psychological_state"]["pressure"] > 0.0
    assert len(treated_target["episodic_memory"]) == 2
    assert treated_other == control_other


def test_restored_psychology_reaches_next_coach_trigger_and_changes_fallback_plan():
    source = _agent("Alpha")
    source.psychological_state = PsychologicalState(
        morale=0.0,
        pressure=0.0,
        trust=1.0,
        risk_appetite=1.0,
        conflict=0.0,
        audience_activation=0.0,
    )
    state = capture_society_continuity(
        source, source_transaction_id="season-1:fixture-1",
    )
    restored = _agent("Alpha")
    restored.squad_carryover = TeamSquadCarryover(
        team_id="Alpha", society_state=state,
    )
    apply_carryover_to_agent(restored)
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True), llm=None, use_llm=False,
    )
    trigger = CognitiveTriggerEvent(
        60.0,
        "goal_conceded",
        ENTITY_TIER_COACH,
        "coach:Alpha",
        team_id="Alpha",
        salience=1.0,
    )

    record = executor.process_trigger(
        trigger, _match_state(), home_agent=restored, away_agent=_agent("Beta"),
    )

    context = trigger.facts["society_continuity_context"]
    assert context["source_state_identity"] == state["state_identity"]
    assert context["memory_text_authority"] == "untrusted_context_only"
    assert record.plan["controls_delta"]["risk_budget"] > -0.03
    assert record.applied is True


def test_malformed_continuity_modifiers_cannot_break_offline_fallback():
    executor = CognitiveExecutor(
        CognitiveMatchConfig(enabled=True), llm=None, use_llm=False,
    )
    trigger = CognitiveTriggerEvent(
        60.0,
        "goal_conceded",
        ENTITY_TIER_COACH,
        "coach:Alpha",
        team_id="Alpha",
        salience=1.0,
        facts={
            "society_continuity_context": {
                "psychological_decision_modifiers": {
                    "risk": "invalid",
                    "coordination": float("inf"),
                },
            },
        },
    )

    record = executor.process_trigger(trigger, _match_state())

    assert record.plan["controls_delta"] == {
        "pressing_intensity": -0.06,
        "risk_budget": -0.03,
    }


def test_society_state_rejects_identity_semantic_and_size_tampering():
    state = capture_society_continuity(
        _agent("Alpha"), source_transaction_id="season-1:fixture-1",
    )
    identity_tamper = copy.deepcopy(state)
    identity_tamper["tactical_controls"]["risk_budget"] = 0.9
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_society_continuity_state(identity_tamper, expected_team="Alpha")

    semantic_tamper = copy.deepcopy(state)
    semantic_tamper["scalars"]["referee_grievance"] = 2.0
    _rehash(semantic_tamper)
    with pytest.raises(ValueError, match="out of bounds"):
        validate_society_continuity_state(semantic_tamper, expected_team="Alpha")

    oversized = copy.deepcopy(state)
    oversized["reflection_diary"] = "x" * 2049
    _rehash(oversized)
    with pytest.raises(ValueError, match="string exceeds limit"):
        validate_society_continuity_state(oversized, expected_team="Alpha")

    invalid_keys = copy.deepcopy(state)
    invalid_keys["semantic_memory"] = {"valid": 1, 2: "invalid"}
    with pytest.raises(ValueError, match="mapping key is invalid"):
        validate_society_continuity_state(invalid_keys, expected_team="Alpha")

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_society_continuity_state(state, expected_team="Beta")

    with pytest.raises(ValueError, match="identity mismatch"):
        capture_society_continuity(_agent("Alpha"), source_transaction_id="")

    meta_agent = _agent("Alpha")
    meta_agent.apply_reflection_payload({
        "reflection": "Keep this proposal in shadow.",
        "suggested_adjustments": {"risk_budget": -0.04},
    }, operation_id="reflection:tamper")
    nested_tamper = capture_society_continuity(
        meta_agent, source_transaction_id="season-1:fixture-1",
    )
    nested_tamper["reflection_audit"][0]["meta_proposal"]["changes"][
        "risk_budget"
    ]["after"] = 0.99
    _rehash(nested_tamper)
    with pytest.raises(
        ValueError, match="change is out of bounds|proposal identity",
    ):
        validate_society_continuity_state(
            nested_tamper, expected_team="Alpha",
        )


def test_public_society_snapshot_never_exposes_memory_text():
    agent = _agent("Alpha")
    agent._register_memory_event(
        "PRIVATE NARRATIVE CONTENT", event_type="micro_cognitive",
    )
    state = capture_society_continuity(
        agent, source_transaction_id="season-1:fixture-1",
    )

    snapshot = society_public_snapshot(state, team_id="Alpha")

    assert snapshot["available"] is True
    assert snapshot["memory_records"] == 1
    assert snapshot["cognitive_memory_records"] == 1
    assert "PRIVATE NARRATIVE CONTENT" not in json.dumps(snapshot)


def test_public_society_transition_reports_creation_and_removal():
    unavailable = society_public_snapshot(None, team_id="Alpha")
    state = capture_society_continuity(
        _agent("Alpha"), source_transaction_id="season-1:fixture-1",
    )
    available = society_public_snapshot(state, team_id="Alpha")

    created = society_public_transition(unavailable, available)
    removed = society_public_transition(available, unavailable)

    assert created["available"] is True
    assert created["before_available"] is False
    assert created["after_available"] is True
    assert removed["available"] is True
    assert removed["before_available"] is True
    assert removed["after_available"] is False

    invalid = copy.deepcopy(available)
    invalid["memory_records"] = 97
    with pytest.raises(ValueError, match="count exceeds limit"):
        validate_society_public_snapshot(invalid)


def test_match_settlement_captures_society_state_and_is_cognitively_idempotent(
    tmp_path,
):
    home = _agent("Alpha")
    away = _agent("Beta")
    summary = SimpleNamespace(
        cognitive_plans=[{
            "trigger": {
                "kind": "goal_conceded",
                "team_id": "Alpha",
                "entity_tier": "coach",
                "entity_id": "coach:Alpha",
                "salience": 0.9,
            },
            "plan": {
                "reasoning": "Retain compactness and reduce transition risk.",
            },
            "applied": True,
        }],
        player_stats={},
        cognitive_triggers=[],
        cognitive_tier_usage={"coach": 1},
    )
    kwargs = {
        "base_dir": str(tmp_path),
        "result_home": "draw",
        "result_away": "draw",
        "score_diff_home": 0,
        "xg_home": 1.0,
        "xg_away": 1.0,
        "prof_score": 0.0,
        "social_chaos": 0.0,
        "stage_name": "md-01",
        "micro_summary": summary,
        "transaction_id": "season-1:fixture-1",
    }

    finalize_match_feedback(home, away, **kwargs)
    first_state = copy.deepcopy(home.squad_carryover.society_state)
    first_memory_count = len(home.episodic_memory)
    finalize_match_feedback(home, away, **kwargs)

    assert home.squad_carryover.society_state == first_state
    assert len(home.episodic_memory) == first_memory_count == 1
    assert away.episodic_memory == []
    payload = json.loads((
        tmp_path / "data/persistence/squad_carryover.json"
    ).read_text(encoding="utf-8"))
    assert payload["Alpha"]["society_state"]["state_identity"] == (
        first_state["state_identity"]
    )


def test_cognitive_runtime_packet_is_compacted_before_cross_match_persistence():
    agent = _agent("Alpha")
    agent.ingest_micro_cognitive_memory([{
        "trigger": {
            "kind": "xg_swing",
            "team_id": "Alpha",
            "entity_tier": "coach",
            "entity_id": "coach:Alpha",
            "salience": 0.8,
            "facts": {"world_model_decision_support": "x" * 200_000},
        },
        "plan": {"reasoning": "Use a bounded response to the current swing."},
        "applied": True,
    }], "Alpha", operation_id="season-1:fixture-1")

    state = capture_society_continuity(
        agent, source_transaction_id="season-1:fixture-1",
    )

    encoded = json.dumps(state, ensure_ascii=False, allow_nan=False)
    metadata = state["episodic_memory"][0]["metadata"]
    assert len(encoded.encode("utf-8")) < 262_144
    assert metadata == {
        "applied": True,
        "continuity_event_id": "season-1:fixture-1:0",
    }
    assert "world_model_decision_support" not in encoded
