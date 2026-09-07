from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from src.simulation.agent import SocietyAgent
from src.simulation.cross_match_state import (
    TeamSquadCarryover,
    apply_carryover_to_agent,
)
from src.simulation.match_pipeline import finalize_match_feedback
from src.simulation.meta_learning import (
    MetaLearningController,
    build_meta_evaluation_receipt,
    observe_agent_meta_proposals,
    validate_meta_evaluation_receipt,
    validate_meta_proposal,
)


def _agent(name: str) -> SocietyAgent:
    return SocietyAgent(
        name,
        {
            "tier": "Semi-Core",
            "final_status_score": 55.0,
            "c1_win_rate": 50.0,
            "c3_major_exp": 40.0,
            "c5_pressure": 45.0,
        },
        random_root_seed=317,
    )


def _reflection(
    agent: SocietyAgent,
    operation_id: str = "reflection:1",
    *,
    expires_after: int = 2,
):
    evidence = agent._register_memory_event(
        "A verified prior tactical lesson",
        event_type="decision_event",
        source="match_log",
    )
    audit = agent.apply_reflection_payload({
        "reflection": "Consider a slightly lower transition risk.",
        "confidence": 1.0,
        "evidence_memory_ids": [evidence["id"]],
        "suggested_adjustments": {"risk_budget": -0.08},
        "expires_after": expires_after,
    }, operation_id=operation_id)
    return audit


def _settle(
    root,
    home: SocietyAgent,
    away: SocietyAgent,
    transaction_id: str,
) -> None:
    finalize_match_feedback(
        home,
        away,
        base_dir=str(root),
        result_home="win",
        result_away="loss",
        score_diff_home=1,
        xg_home=1.4,
        xg_away=0.8,
        prof_score=0.1,
        social_chaos=0.0,
        stage_name="md-02",
        micro_summary=SimpleNamespace(
            cognitive_plans=[],
            player_stats={},
            cognitive_triggers=[],
            cognitive_tier_usage={},
        ),
        transaction_id=transaction_id,
    )


def _matched_rows(effect: float, *, count: int = 8):
    return [{
        "unit_id": f"unit-{index}",
        "seed": index,
        "control_utility": 0.0,
        "treated_utility": effect,
        "control_source_identity": f"{index + 1:064x}",
        "treated_source_identity": f"{index + 101:064x}",
    } for index in range(count)]


def test_reflection_stages_identity_bound_shadow_without_parameter_authority():
    first = _agent("Alpha")
    second = _agent("Alpha")
    before = first.tactical_controls["risk_budget"]

    first_audit = _reflection(first)
    second_audit = _reflection(second)

    assert first_audit["proposal_status"] == "shadow"
    assert first_audit["authorization_required"] is True
    assert first_audit["applied_adjustments"] == {}
    assert first.tactical_controls["risk_budget"] == before
    assert first_audit["proposal_identity"] == second_audit["proposal_identity"]
    proposal = validate_meta_proposal(first_audit["meta_proposal"])
    assert proposal.agent == "Alpha"
    assert proposal.evidence_ids


def test_post_match_observation_is_noncausal_idempotent_and_cross_match(
    tmp_path,
):
    home = _agent("Alpha")
    away = _agent("Beta")
    audit = _reflection(home)
    before = home.tactical_controls["risk_budget"]

    _settle(tmp_path, home, away, "season-1:md02")
    first_state = copy.deepcopy(home.squad_carryover.society_state)
    _settle(tmp_path, home, away, "season-1:md02")

    assert audit["proposal_status"] == "observed_pending_evaluation"
    assert audit["observation_count"] == 1
    proposal = validate_meta_proposal(audit["meta_proposal"])
    assert len(proposal.observations) == 1
    assert proposal.observations[0]["claim_boundary"].endswith(
        "not_effect_evidence"
    )
    assert home.tactical_controls["risk_budget"] == before
    assert home.squad_carryover.society_state == first_state

    restored = _agent("Alpha")
    restored.squad_carryover = TeamSquadCarryover.from_dict(
        home.squad_carryover.to_dict()
    )
    apply_carryover_to_agent(restored)
    restored_audit = restored.llm_reflection_audit[0]
    assert restored_audit["proposal_identity"] == audit["proposal_identity"]
    assert restored_audit["proposal_status"] == "observed_pending_evaluation"
    assert restored.tactical_controls["risk_budget"] == before


def test_matched_evaluation_receipt_is_required_and_idempotently_authorizes():
    agent = _agent("Alpha")
    audit = _reflection(agent)
    proposal = validate_meta_proposal(audit["meta_proposal"])
    before = agent.tactical_controls["risk_budget"]
    receipt = build_meta_evaluation_receipt(
        proposal,
        matched_rows=_matched_rows(0.04, count=12),
    )
    assert validate_meta_evaluation_receipt(
        receipt,
        proposal_identity=proposal.proposal_identity,
        agent=proposal.agent,
    ) == receipt
    integer_minimum_receipt = build_meta_evaluation_receipt(
        proposal,
        matched_rows=_matched_rows(0.04, count=12),
        minimum_effect=0,
    )
    assert integer_minimum_receipt["minimum_effect"] == 0.0

    committed = agent.authorize_meta_proposal(
        proposal.proposal_id, receipt,
    )
    after_first = agent.tactical_controls["risk_budget"]
    repeated = agent.authorize_meta_proposal(
        proposal.proposal_id, receipt,
    )

    assert committed["proposal_status"] == "committed"
    assert after_first < before
    assert agent.tactical_controls["risk_budget"] == after_first
    assert repeated["authorization_identity"] == receipt["receipt_identity"]
    validate_meta_proposal(repeated["meta_proposal"])


def test_nonpositive_interval_rejects_without_mutation():
    agent = _agent("Alpha")
    audit = _reflection(agent)
    proposal = validate_meta_proposal(audit["meta_proposal"])
    before = agent.tactical_controls["risk_budget"]
    receipt = build_meta_evaluation_receipt(
        proposal,
        matched_rows=[
            {
                **row,
                "treated_utility": effect,
            }
            for row, effect in zip(
                _matched_rows(0.0),
                (-0.08, -0.05, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08),
            )
        ],
    )

    rejected = agent.authorize_meta_proposal(
        proposal.proposal_id, receipt,
    )

    assert rejected["proposal_status"] == "rejected"
    assert rejected["applied_adjustments"] == {}
    assert agent.tactical_controls["risk_budget"] == before


def test_proposal_and_evaluation_tampering_fail_closed():
    agent = _agent("Alpha")
    audit = _reflection(agent)
    proposal = validate_meta_proposal(audit["meta_proposal"])

    tampered_proposal = copy.deepcopy(audit["meta_proposal"])
    tampered_proposal["changes"]["risk_budget"]["after"] = 0.99
    with pytest.raises(ValueError, match="change is out of bounds|identity"):
        validate_meta_proposal(tampered_proposal)

    with pytest.raises(ValueError, match="matched rows"):
        build_meta_evaluation_receipt(
            proposal,
            matched_rows=_matched_rows(0.04, count=7),
        )
    with pytest.raises(ValueError, match="receipt is invalid"):
        build_meta_evaluation_receipt(
            proposal,
            matched_rows=_matched_rows(0.04),
            uses_sealed_data=True,
        )
    with pytest.raises(ValueError, match="receipt is invalid"):
        build_meta_evaluation_receipt(
            proposal,
            matched_rows=_matched_rows(0.04),
            minimum_effect=-0.01,
        )
    out_of_range = _matched_rows(0.04)
    out_of_range[0]["treated_utility"] = 5.01
    with pytest.raises(ValueError, match="utility is out of bounds"):
        build_meta_evaluation_receipt(
            proposal,
            matched_rows=out_of_range,
        )

    receipt = build_meta_evaluation_receipt(
        proposal,
        matched_rows=_matched_rows(0.04),
    )
    tampered_receipt = copy.deepcopy(receipt)
    tampered_receipt["ci_low"] = 0.03
    with pytest.raises(ValueError, match="receipt is invalid"):
        agent.authorize_meta_proposal(
            proposal.proposal_id, tampered_receipt,
        )


def test_parameter_drift_blocks_authorization_without_partial_mutation():
    agent = _agent("Alpha")
    audit = _reflection(agent)
    proposal = validate_meta_proposal(audit["meta_proposal"])
    receipt = build_meta_evaluation_receipt(
        proposal,
        matched_rows=_matched_rows(0.04),
    )
    agent.tactical_controls["risk_budget"] = 0.7

    with pytest.raises(RuntimeError, match="changed since proposal"):
        agent.authorize_meta_proposal(proposal.proposal_id, receipt)

    assert agent.tactical_controls["risk_budget"] == 0.7
    assert audit["proposal_status"] == "shadow"
    assert audit["applied_adjustments"] == {}


def test_pending_proposal_expires_after_observation_window(tmp_path):
    home = _agent("Alpha")
    away = _agent("Beta")
    audit = _reflection(home, expires_after=1)

    _settle(tmp_path, home, away, "season-1:md02")
    assert audit["proposal_status"] == "observed_pending_evaluation"
    _settle(tmp_path, home, away, "season-1:md03")

    assert audit["proposal_status"] == "expired"
    assert audit["observation_count"] == 1
    assert home.tactical_controls["risk_budget"] == 0.5


def test_operation_identity_is_idempotent_but_cannot_be_reused_for_new_content():
    agent = _agent("Alpha")
    controller = MetaLearningController()
    reflection = {
        "reflection": "One bounded proposal",
        "suggested_adjustments": {"risk_budget": -0.02},
    }

    first = controller.propose(
        agent, reflection, operation_id="reflection:stable",
    )
    repeated = controller.propose(
        agent, reflection, operation_id="reflection:stable",
    )
    assert repeated.proposal_identity == first.proposal_identity

    changed = {
        **reflection,
        "suggested_adjustments": {"risk_budget": 0.02},
    }
    with pytest.raises(ValueError, match="identity was reused"):
        controller.propose(
            agent, changed, operation_id="reflection:stable",
        )


def test_empty_reflection_is_audited_without_creating_meta_authority():
    agent = _agent("Alpha")
    audit = agent.apply_reflection_payload({
        "reflection": "No parameter adjustment is supported.",
        "confidence": 0.8,
        "suggested_adjustments": {},
    }, operation_id="reflection:no-change")

    assert audit["proposal_status"] == "not_proposed"
    assert audit["authorization_required"] is False
    assert audit["proposed_adjustments"] == {}
    assert "meta_proposal" not in audit
    assert observe_agent_meta_proposals(
        agent,
        operation_id="season-1:md02",
        result_utility=0.5,
        goal_difference=1.0,
        xg_difference=0.4,
    ) == []

    proposal_audit = _reflection(agent, operation_id="reflection:bounded")
    proposal = validate_meta_proposal(proposal_audit["meta_proposal"])
    with pytest.raises(ValueError, match="observation utility is out of bounds"):
        MetaLearningController().observe(
            proposal,
            operation_id="season-1:md03",
            result_utility=5.01,
            goal_difference=1.0,
            xg_difference=0.4,
        )

    invalid = agent.apply_reflection_payload({
        "reflection": "Boolean values are not numeric authority.",
        "confidence": True,
        "suggested_adjustments": {"risk_budget": True},
    }, operation_id="reflection:boolean-adjustment")
    assert invalid["proposal_status"] == "not_proposed"
    assert invalid["confidence"] == 0.5
    assert invalid["rejected_adjustments"] == {
        "risk_budget": "non_numeric_delta",
    }

    valid_proposal = validate_meta_proposal(
        _reflection(agent, operation_id="reflection:typed-observation")[
            "meta_proposal"
        ]
    )
    with pytest.raises(ValueError, match="must be numeric"):
        MetaLearningController().observe(
            valid_proposal,
            operation_id="season-1:md04",
            result_utility=True,
            goal_difference=1.0,
            xg_difference=0.4,
        )


def test_boolean_lifecycle_values_fail_closed():
    agent = _agent("Alpha")
    controller = MetaLearningController()
    reflection = {
        "reflection": "Invalid lifecycle value.",
        "suggested_adjustments": {"risk_budget": -0.02},
        "expires_after": True,
    }

    with pytest.raises(ValueError, match="expiry must be an integer"):
        controller.propose(
            agent, reflection, operation_id="reflection:boolean-expiry",
        )
