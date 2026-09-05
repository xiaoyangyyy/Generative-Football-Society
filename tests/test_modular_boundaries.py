"""Regression tests for the boundaries extracted from former god objects."""

import ast
import inspect
import random
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from src.simulation.agent import SocietyAgent
from src.simulation.agent_dynamics import AgentMatchDynamicsMixin
from src.simulation.agent_initialization import AgentInitializationMixin
from src.simulation.agent_memory import AgentMemoryMixin
from src.simulation.agent_memory_beliefs import AgentBeliefMemoryMixin
from src.simulation.agent_memory_retrieval import AgentMemoryRetrievalMixin
from src.simulation.agent_memory_write import AgentMemoryWriteMixin
from src.simulation.agent_psychology import AgentPsychologyMixin
from src.simulation.agent_social import SocialAgentMixin
from src.simulation.referee_policy import RefereePolicy, normalize_weights
from src.simulation.tournament_2026 import TournamentManager
from src.simulation.tournament_finalize import TournamentFinalizeMixin
from src.simulation.tournament_match import TournamentMatchMixin
from src.simulation.tournament_reporting import TournamentReportingMixin
from src.simulation.tournament_scoring import TournamentScoringMixin
from src.simulation.tournament_setup import TournamentSetupMixin


def test_social_behaviour_is_inherited_from_dedicated_mixin():
    assert issubclass(SocietyAgent, SocialAgentMixin)
    assert "apply_social_signal" not in SocietyAgent.__dict__
    assert SocietyAgent.apply_social_signal is SocialAgentMixin.apply_social_signal


def test_memory_behaviour_is_inherited_from_dedicated_mixin():
    assert issubclass(SocietyAgent, AgentMemoryMixin)
    assert "_register_memory_event" not in SocietyAgent.__dict__
    assert SocietyAgent._register_memory_event is AgentMemoryMixin._register_memory_event


def test_memory_composition_preserves_subsystem_ownership():
    assert all(
        issubclass(AgentMemoryMixin, mixin)
        for mixin in (
            AgentMemoryWriteMixin,
            AgentMemoryRetrievalMixin,
            AgentBeliefMemoryMixin,
        )
    )
    assert (
        AgentMemoryMixin._register_memory_event
        is AgentMemoryWriteMixin._register_memory_event
    )
    assert (
        AgentMemoryMixin.retrieve_memory_context
        is AgentMemoryRetrievalMixin.retrieve_memory_context
    )
    assert (
        AgentMemoryMixin.update_beliefs_from_match_event
        is AgentBeliefMemoryMixin.update_beliefs_from_match_event
    )


def test_memory_registration_stays_decomposed():
    source_lines, _ = inspect.getsourcelines(
        AgentMemoryWriteMixin._register_memory_event
    )
    assert len(source_lines) <= 55
    assert {
        "_memory_write_diagnostics",
        "_build_memory_record",
        "_store_memory_record",
    }.issubset(AgentMemoryWriteMixin.__dict__)


def test_match_dynamics_are_inherited_and_stage_driven():
    assert issubclass(SocietyAgent, AgentMatchDynamicsMixin)
    assert (
        SocietyAgent.recursive_update
        is AgentMatchDynamicsMixin.recursive_update
    )
    source_lines, _ = inspect.getsourcelines(
        AgentMatchDynamicsMixin.recursive_update
    )
    assert len(source_lines) <= 30
    assert {
        "_build_match_update_context",
        "_update_match_latent_state",
        "_record_match_outcome",
    }.issubset(AgentMatchDynamicsMixin.__dict__)


def test_agent_initialization_is_ordered_and_decomposed():
    assert issubclass(SocietyAgent, AgentInitializationMixin)
    source = textwrap.dedent(inspect.getsource(SocietyAgent.__init__))
    method = ast.parse(source).body[0]
    assert isinstance(method, ast.FunctionDef)
    assert len(method.body) <= 7
    assert {
        "_initialize_identity",
        "_initialize_roles_and_strategy",
        "_initialize_runtime_state",
        "_initialize_affective_state",
    }.issubset(AgentInitializationMixin.__dict__)


def test_agent_initialization_preserves_required_state_contract():
    agent = SocietyAgent(
        "Test United",
        {
            "tier": "Semi-Core",
            "final_status_score": 55.0,
            "c1_win_rate": 52.0,
            "c3_major_exp": 45.0,
            "c5_pressure": 40.0,
        },
    )
    assert agent.team_name == "Test United"
    assert set(agent.tactical_controls) == {
        "pressing_intensity",
        "risk_budget",
        "line_height",
        "rotation_aggressiveness",
    }
    assert set(agent.z_state) == {
        "morale",
        "stability",
        "unity",
        "confidence",
        "risk_tolerance",
        "pressing_intensity",
        "referee_trust",
        "media_sensitivity",
    }
    assert agent.episodic_memory == []
    assert agent.memory_event_log == []


def test_agent_psychology_is_inherited_from_dedicated_mixin():
    assert issubclass(SocietyAgent, AgentPsychologyMixin)
    for method_name in {
        "_initialize_psychology_from_history",
        "_initialize_latent_states",
        "_project_latents_to_states",
        "_appraise_event",
        "_emotion_from_appraisal",
        "_coping_from_appraisal_emotion",
        "_memory_salience",
    }:
        assert method_name not in SocietyAgent.__dict__
        assert getattr(SocietyAgent, method_name) is getattr(
            AgentPsychologyMixin, method_name
        )
    assert SocietyAgent.morale is AgentPsychologyMixin.morale
    assert SocietyAgent.hidden_state is AgentPsychologyMixin.hidden_state


def test_agent_psychology_projection_remains_finite_and_normalized():
    agent = SocietyAgent(
        "Psychology FC",
        {
            "tier": "Semi-Core",
            "final_status_score": 57.0,
            "c1_win_rate": 51.0,
            "c3_major_exp": 43.0,
            "c5_pressure": 48.0,
        },
        random_root_seed=77,
    )
    appraisal = agent._appraise_event(
        {
            "result": "loss",
            "score_diff": -1.0,
            "stage_pressure": 0.8,
            "referee_controversy": 0.4,
            "upset_factor": 0.2,
            "social_chaos": 0.3,
        }
    )
    emotion = agent._emotion_from_appraisal(appraisal)
    coping = agent._coping_from_appraisal_emotion(appraisal, emotion)

    assert np.isfinite(agent.hidden_state).all()
    assert np.isclose(sum(emotion.values()), 1.0)
    assert np.isclose(
        coping["planning"]
        + coping["self_correction"]
        + coping["external_blame"],
        1.0,
    )
    assert -1.0 <= coping["risk_shift"] <= 1.0


def test_agent_local_initialization_rng_is_replayable_and_does_not_touch_global_rng():
    stats = {
        "tier": "Semi-Core", "final_status_score": 55.0,
        "c1_win_rate": 52.0, "c3_major_exp": 45.0, "c5_pressure": 40.0,
    }
    random.seed(991)
    expected_next = random.random()
    random.seed(991)
    first = SocietyAgent(
        "Seeded United", stats, initialization_rng=random.Random(17),
    )
    observed_next = random.random()
    second = SocietyAgent(
        "Seeded United", stats, initialization_rng=random.Random(17),
    )

    assert observed_next == expected_next
    assert first.personality == second.personality
    assert first.coach_authority == second.coach_authority
    assert first.referee_trust == second.referee_trust


def test_agent_default_initialization_is_identity_scoped():
    stats = {"tier": "Core", "final_status_score": 65.0}
    random.seed(117)
    expected_next = random.random()
    random.seed(117)
    first = SocietyAgent("Identity FC", stats, random_root_seed=91)
    observed_next = random.random()
    second = SocietyAgent("Identity FC", stats, random_root_seed=91)

    assert observed_next == expected_next
    assert first.personality == second.personality


def test_match_execution_is_inherited_from_dedicated_mixin():
    assert issubclass(TournamentManager, TournamentMatchMixin)
    assert "play_match" not in TournamentManager.__dict__
    assert TournamentManager.play_match is TournamentMatchMixin.play_match


def test_match_orchestrator_stays_small_and_stage_driven():
    source_lines, _ = inspect.getsourcelines(TournamentMatchMixin.play_match)
    assert len(source_lines) <= 80
    assert {
        "_prepare_match",
        "_simulate_match",
        "_run_narrative_and_social",
        "_finalize_match_state",
    }.issubset(dir(TournamentMatchMixin))
    assert all(
        issubclass(TournamentMatchMixin, mixin)
        for mixin in (
            TournamentSetupMixin,
            TournamentScoringMixin,
            TournamentReportingMixin,
            TournamentFinalizeMixin,
        )
    )


def test_match_preparation_stays_stage_driven():
    source_lines, _ = inspect.getsourcelines(TournamentSetupMixin._prepare_match)
    assert len(source_lines) <= 55
    assert {
        "_resolve_match_participants",
        "_prepare_team_recovery",
        "_build_coaching_contexts",
        "_apply_match_tactics",
        "_resolve_internal_and_referee_game",
    }.issubset(TournamentSetupMixin.__dict__)


def test_numeric_match_stage_stays_decomposed():
    source_lines, _ = inspect.getsourcelines(TournamentMatchMixin._simulate_match)
    assert len(source_lines) <= 110
    assert {
        "_build_fused_match_context",
        "_resolve_regulation_score",
        "_resolve_knockout_score",
        "_report_match_result",
    }.issubset(dir(TournamentMatchMixin))
    assert (
        TournamentMatchMixin._prepare_match
        is TournamentSetupMixin._prepare_match
    )
    assert (
        TournamentMatchMixin._resolve_regulation_score
        is TournamentScoringMixin._resolve_regulation_score
    )
    assert (
        TournamentMatchMixin._report_match_result
        is TournamentReportingMixin._report_match_result
    )
    assert (
        TournamentMatchMixin._finalize_match_state
        is TournamentFinalizeMixin._finalize_match_state
    )


def test_regulation_scoring_stays_stage_driven():
    source_lines, _ = inspect.getsourcelines(
        TournamentScoringMixin._resolve_regulation_score
    )
    assert len(source_lines) <= 75
    assert {
        "_build_regulation_context",
        "_run_physics_official_score",
        "_run_macro_regulation_score",
    }.issubset(TournamentScoringMixin.__dict__)


def test_match_finalization_stays_stage_driven():
    source_lines, _ = inspect.getsourcelines(
        TournamentFinalizeMixin._finalize_match_state
    )
    assert len(source_lines) <= 50
    assert {
        "_update_post_match_agents",
        "_apply_physical_wear",
        "_record_match_decisions",
        "_report_locker_room_state",
        "_commit_match_result",
    }.issubset(TournamentFinalizeMixin.__dict__)


def test_reporting_pipeline_stays_stage_driven():
    source_lines, _ = inspect.getsourcelines(
        TournamentReportingMixin._run_narrative_and_social
    )
    assert len(source_lines) <= 50
    assert {
        "_build_facts_ledger",
        "_apply_narrative_atmosphere",
        "_publish_media_matrix",
        "_run_post_match_dialogue",
    }.issubset(TournamentReportingMixin.__dict__)


def test_result_reporting_stays_decomposed():
    source_lines, _ = inspect.getsourcelines(
        TournamentReportingMixin._report_match_result
    )
    assert len(source_lines) <= 50
    assert {
        "_build_result_payload",
        "_build_xg_context",
        "_audit_match_debug",
        "_run_report_replay",
    }.issubset(TournamentReportingMixin.__dict__)


def test_reporting_stage_contexts_reject_missing_and_unknown_fields():
    with pytest.raises(TypeError, match="missing="):
        TournamentReportingMixin._report_match_result(object())
    with pytest.raises(TypeError, match="unexpected="):
        TournamentReportingMixin._report_match_result(object(), unknown=True)
    with pytest.raises(TypeError, match="missing="):
        TournamentReportingMixin._run_narrative_and_social(object())
    with pytest.raises(TypeError, match="unexpected="):
        TournamentReportingMixin._run_narrative_and_social(object(), unknown=True)


def test_referee_weights_are_normalized_without_manager_state():
    weights = normalize_weights({"lenient": 2.0, "strict": 1.0}, ["lenient", "strict"])
    assert weights == {"lenient": 2.0 / 3.0, "strict": 1.0 / 3.0}


def test_referee_policy_produces_a_valid_stage_distribution_and_sample():
    policy = RefereePolicy(stage_morph_strength=1.0)
    group = policy.stage_distribution(0.0)
    final = policy.stage_distribution(1.0)
    assert np.isclose(sum(group.values()), 1.0)
    assert np.isclose(sum(final.values()), 1.0)
    assert final["strict"] > group["strict"]

    first = SimpleNamespace(media_exposure=0.8)
    second = SimpleNamespace(media_exposure=0.2)
    sample = policy.sample(
        first, second, 0.7, rng=np.random.default_rng(42),
    )
    assert sample["profile_name"] in policy.profiles
    assert 0.05 <= sample["strictness"] <= 0.98
    assert sample["bias_t2"] == -sample["bias_t1"]
