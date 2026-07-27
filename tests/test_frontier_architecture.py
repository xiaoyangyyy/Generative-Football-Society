from types import SimpleNamespace

import numpy as np

from src.match_engine.event_clock import CompetingRiskClock
from src.match_engine.hierarchical_policy import HierarchicalPolicy
from src.match_engine.world_model.active_sampling import ActiveSamplingQueue
from src.match_engine.world_model.graph import build_interaction_graph
from src.match_engine.world_model.observation import OBS_DIM
from src.simulation.counterfactual import evaluate_intervention
from src.simulation.meta_learning import MetaLearningController
from src.simulation.runtime import SimulationConfig, build_run_manifest, environment_snapshot, write_manifest
from src.simulation.agent import SocietyAgent
from src.data_engine.dataset_registry import files_for_split, stable_partition
from src.simulation.standings import apply_group_result
from src.match_engine.receiver_ranker import ReceiverRanker


def test_paired_counterfactual_recovers_exact_effect():
    def run(seed, treatment):
        shared_noise = np.random.default_rng(seed).normal()
        return shared_noise + float(treatment)

    report = evaluate_intervention(run, 0.0, 0.25, samples=16)
    assert np.isclose(report.average_treatment_effect, 0.25)
    assert report.directionally_supported


def test_runtime_manifest_hash_and_atomic_write(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a\n1\n", encoding="utf-8")
    cfg = SimulationConfig(seed=7)
    manifest = build_run_manifest(cfg, tmp_path, data_paths=[data])
    target = write_manifest(tmp_path / "run.json", manifest)
    assert target.is_file()
    assert manifest["root_seed"] == 7
    assert len(manifest["artifacts"]["data"][0]["sha256"]) == 64
    values = environment_snapshot({"MATCH_MICRO": "0", "GFS_SEED": "9"})
    assert SimulationConfig.from_mapping(values).seed == 9
    with np.testing.assert_raises(TypeError):
        values["GFS_SEED"] = "10"


def test_dataset_split_is_stable_and_sealed_test_is_isolated():
    assert stable_partition("match-a", 42) == stable_partition("match-a", 42)
    manifest = {"files": [
        {"path": "a.jsonl", "split": "train"},
        {"path": "b.jsonl", "split": "sealed_test"},
    ]}
    assert files_for_split(manifest, {"train"}) == {"a.jsonl"}
    with np.testing.assert_raises(ValueError):
        files_for_split(manifest, {"train", "sealed_test"})


def test_standings_service_is_validated_and_pure_of_manager_state():
    table = {name: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for name in ("A", "B")}
    apply_group_result(table, "A", "B", 2, 1)
    assert table["A"] == {"pts": 3, "gf": 2, "ga": 1, "gd": 1}
    with np.testing.assert_raises(ValueError):
        apply_group_result(table, "A", "B", -1, 0)


def test_receiver_ranker_validates_features_and_scores_finitely():
    ranker = ReceiverRanker(np.zeros(5), np.ones(5), np.arange(5) / 10.0, 0.2)
    assert np.isfinite(ranker.score([0.1, 0.2, 0.3, 0.4, 0.5]))
    with np.testing.assert_raises(ValueError):
        ranker.score([0.1, np.nan, 0.3, 0.4, 0.5])


def test_competing_risk_clock_and_modulation():
    clock = CompetingRiskClock()
    rng = np.random.default_rng(3)
    events = [clock.sample({"pass": 0.8, "shot": 0.2}, rng) for _ in range(2000)]
    shot_rate = sum(event.kind == "shot" for event in events) / len(events)
    assert 0.16 < shot_rate < 0.24
    assert 0.85 < np.mean([event.wait_seconds for event in events]) < 1.15
    changed = clock.modulate({"shot": 0.2}, {"risk": 1.0}, {"shot": {"risk": 1.0}})
    assert changed["shot"] > 0.2


def test_hierarchical_policy_preserves_role_relevance():
    policy = HierarchicalPolicy()
    options = policy.decompose({"pressing_intensity": 0.8, "risk_budget": 0.6})
    front = next(option for option in options if option.unit == "front_line")
    assert policy.player_intent(front, "ST")["risk"] > policy.player_intent(front, "GK")["risk"]


def test_meta_proposal_is_shadow_until_evaluated_and_can_rollback():
    agent = SimpleNamespace(
        name="A", tactical_controls={"risk_budget": 0.5},
        roles={"Icon": {"patience": 0.8}}, W_h=0.6, W_x=0.4,
        memory_event_log=[{"id": "m1"}],
    )
    controller = MetaLearningController()
    proposal = controller.propose(agent, {
        "confidence": 1.0, "evidence_memory_ids": ["m1"],
        "suggested_adjustments": {"risk_budget": 0.1},
    })
    assert agent.tactical_controls["risk_budget"] == 0.5
    controller.commit(agent, proposal)
    assert agent.tactical_controls["risk_budget"] > 0.5
    controller.rollback(agent, proposal)
    assert agent.tactical_controls["risk_budget"] == 0.5


def _full_observation():
    obs = np.full(OBS_DIM, 0.1, dtype=np.float32)
    players = obs[210:298].reshape(22, 4)
    players[:, 0] = np.linspace(0.05, 0.95, 22)
    players[:, 1] = np.tile([0.35, 0.65], 11)
    return obs


def test_graph_and_active_sampling_contracts():
    obs = _full_observation()
    graph = build_interaction_graph(obs, radius=0.3)
    assert graph.nodes.shape == (22, 6)
    assert graph.edge_index.shape[0] == 2
    assert not np.any(graph.edge_index[0] == graph.edge_index[1])

    queue = ActiveSamplingQueue(capacity=2)
    low = queue.add("low", obs, uncertainty=0.01, event_count=100)
    high = queue.add("high", obs, uncertainty=0.8, event_count=1)
    assert high.priority > low.priority
    assert queue.highest(1)[0].payload == "high"


def test_memory_provenance_retrieval_and_delayed_utility():
    agent = SocietyAgent("Test FC", {"final_status_score": 50.0})
    record = agent._register_memory_event(
        "A verified tactical lesson", source="match_log",
        causal_parent_ids=["event-1"], contradicts=["old-memory"],
    )
    assert record["provenance"]["source"] == "match_log"
    assert record["provenance"]["causal_parent_ids"] == ["event-1"]
    retrieved = agent.retrieve_memory_context(top_k=1)
    assert retrieved[0]["id"] == record["id"]
    assert record["retrieval_count"] == 1
    assert agent.record_memory_utility([record["id"]], 0.4) == 1
    assert record["downstream_utility_sum"] == 0.4
