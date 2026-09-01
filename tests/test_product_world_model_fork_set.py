import json
from types import SimpleNamespace

import pytest

from src.product.manager_future import build_manager_future_context
from src.product.match_plan import WorldModelForkSetPlan
from src.product.season import ManagerDecision, SeasonPlan, new_season_state
from src.product.world_model_fork_set import (
    aggregate_fork_set,
    execute_world_model_fork_set,
    project_fork_set_scenario_evidence,
    render_fork_set_html,
    validate_fork_set_scenario_evidence,
)
from src.product.tasks import BackgroundMatchWorker, ProductTaskQueue, TaskConflict


def _comparison(
    plan, branch, *, changed=1, local=1, differences=1, detailed=False,
):
    status = (
        "local_action_divergence_with_descriptive_future_difference"
        if local and differences else
        "local_action_divergence_without_measured_future_difference"
        if local else
        "no_realized_action_divergence"
    )
    comparison = {
        "fixture": {"home": "Brazil", "away": "Argentina", "seed": plan.seed},
        "eligibility": {
            "eligible_for_world_model_policy_attribution": True,
        },
        "intervention": {
            "scope": "world_model_action_policy",
            "changed_sides": [],
            "baseline_policy": "predict_only",
            "treatment_policy": "action_policy",
            "baseline_tactics": {
                "home": plan.home_tactic, "away": plan.away_tactic,
            },
            "treatment_tactics": {
                "home": plan.home_tactic, "away": plan.away_tactic,
            },
            "branch_at_sec": branch,
            "branch_anchor": {
                "verified": True, "state_identity": "a" * 64,
            },
        },
        "policy_propagation": {
            "available": True,
            "summary": {
                "valid_changed_decisions": changed,
                "locally_attributable_changes": local,
            },
        },
        "counterfactual_future_summary": {
            "available": True, "status": status,
            "summary": {
                "descriptive_outcome_difference_count": differences,
            },
            "claim_authority": {
                "simulator_local_action_attribution": bool(local),
            },
        },
    }
    if detailed:
        decisions = []
        for index in range(changed):
            decisions.append({
                "opportunity_id": f"future:{branch}:{index}",
                "team": "Brazil",
                "t_sec": float(branch + 10 + index),
                "clock": (
                    f"{int((branch + 10 + index) // 60)}:"
                    f"{int((branch + 10 + index) % 60):02d}"
                ),
                "baseline_action": "hold",
                "treatment_action": "pass",
                "recommended_action": "pass",
                "probability_policy_version": "validated_action_simplex_v2",
                "signal_mode": "direct_preference",
                "primary_signal_action": "pass",
                "hold_reference_redistributed": False,
                "hold_reference_probability_delta": 0.0,
                "directly_observed": True,
                "local_policy_attribution_eligible": index < local,
                "downstream_windows": {
                    f"{duration}s": {
                        "delta": {
                            "actions": 0, "passes": -1, "crosses": 1,
                            "shots": 1,
                            "goals": 0, "turnovers": 1,
                        },
                        "additional_policy_changes": index,
                        "causal_attribution_authorized": False,
                    }
                    for duration in (30, 120)
                },
            })
        comparison["policy_propagation"]["summary"].update({
            "decisions_truncated": False,
        })
        comparison["policy_propagation"]["decisions"] = decisions
    return comparison


def test_fork_set_plan_is_fixed_ordered_and_research_only():
    plan = WorldModelForkSetPlan(
        "balanced", "low_block_counter", 42, (1800, 2700, 3600),
    )
    assert plan.as_dict()["fixed_scenario_budget"] == 3
    assert plan.as_dict()["analysis_policy"] == (
        "descriptive_timing_sensitivity_no_ranking"
    )
    assert plan.fork_plan(2700).treatment_plan().world_model_policy == "action_policy"
    plan.validate_for_mode("research")
    with pytest.raises(ValueError, match="research mode"):
        plan.validate_for_mode("stable")
    with pytest.raises(ValueError, match="unique and increasing"):
        WorldModelForkSetPlan("balanced", "balanced", 1, (2700, 1800))
    with pytest.raises(ValueError, match="2 to 4"):
        WorldModelForkSetPlan("balanced", "balanced", 1, (2700,))


def test_aggregate_reports_timing_sensitivity_without_ranking_or_causal_upgrade():
    plan = WorldModelForkSetPlan("balanced", "low_block_counter", 42, (1800, 2700))
    result = aggregate_fork_set(plan, [
        _comparison(plan, 1800, changed=0, local=0, differences=0),
        _comparison(plan, 2700, changed=2, local=1, differences=3),
    ])
    assert result["aggregate"]["timing_sensitivity_observed"] is True
    assert result["aggregate"]["ranking_performed"] is False
    assert result["aggregate"]["best_branch_time"] is None
    assert result["claim_authority"] == {
        "descriptive_simulator_timing_sensitivity": True,
        "best_time_recommendation": False,
        "match_outcome_causality": False,
        "population_inference": False,
        "real_football_causality": False,
        "promotion_authorized": False,
    }
    scenarios = project_fork_set_scenario_evidence({
        "plan": plan.as_dict(), "rows": result["rows"],
    })
    assert [row["branch_minute"] for row in scenarios] == [30.0, 45.0]
    assert scenarios[1]["simulator_local_action_attribution"] is True
    validate_fork_set_scenario_evidence(
        scenarios, plan.branch_times_sec,
    )
    tampered = json.loads(json.dumps(scenarios))
    tampered[1]["changed_actions"] = 3
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_fork_set_scenario_evidence(
            tampered, plan.branch_times_sec,
        )
    invalid_source = {
        "plan": plan.as_dict(),
        "rows": json.loads(json.dumps(result["rows"])),
    }
    invalid_source["rows"][0]["branch_at_sec"] = None
    with pytest.raises(ValueError, match="source time"):
        project_fork_set_scenario_evidence(invalid_source)
    result.update({
        "fixture": {"home": "Brazil", "away": "Argentina"},
        "fixed_scenario_budget": 2, "scenarios_completed": 2,
        "claim_boundary": plan.as_dict()["claim_boundary"],
    })
    document = render_fork_set_html(result)
    assert "多时点未来分叉" in document
    assert "系统没有挑选最佳时点" in document


def test_aggregate_rejects_tampered_seed_tactics_or_order():
    plan = WorldModelForkSetPlan("balanced", "low_block_counter", 42, (1800, 2700))
    first = _comparison(plan, 1800)
    second = _comparison(plan, 2700)
    first["fixture"]["seed"] = 43
    with pytest.raises(ValueError, match="frozen world-model policy contrast"):
        aggregate_fork_set(plan, [first, second])
    first = _comparison(plan, 1800)
    first["intervention"]["treatment_tactics"]["home"] = "gegenpress"
    with pytest.raises(ValueError, match="frozen world-model policy contrast"):
        aggregate_fork_set(plan, [first, second])
    with pytest.raises(ValueError, match="frozen branch order"):
        aggregate_fork_set(plan, [second, _comparison(plan, 1800)])

    first = _comparison(plan, 1800)
    first["intervention"]["branch_anchor"]["state_identity"] = "not-a-hash"
    first["policy_propagation"]["summary"][
        "valid_changed_decisions"
    ] = float("nan")
    result = aggregate_fork_set(plan, [first, second])
    assert result["rows"][0]["eligible"] is False
    assert result["rows"][0]["future_status"] == "descriptive_only_ineligible"
    assert result["rows"][0]["changed_actions"] == 0


def test_scenario_v2_exposes_bounded_action_chain_and_windows():
    plan = WorldModelForkSetPlan(
        "balanced", "low_block_counter", 42, (1800, 2700),
    )
    result = aggregate_fork_set(plan, [
        _comparison(
            plan, 1800, changed=1, local=1,
            differences=1, detailed=True,
        ),
        _comparison(
            plan, 2700, changed=1, local=0,
            differences=0, detailed=True,
        ),
    ])
    scenarios = project_fork_set_scenario_evidence({
        "plan": plan.as_dict(), "rows": result["rows"],
    })
    first = scenarios[0]
    assert first["schema_version"] == 2
    assert first["mechanism_examples_truncated"] is False
    example = first["mechanism_examples"][0]
    assert example["schema_version"] == 2
    assert example["baseline_action"] == "hold"
    assert example["treatment_action"] == "pass"
    assert example["probability_policy_version"] == (
        "validated_action_simplex_v2"
    )
    assert example["signal_mode"] == "direct_preference"
    assert example["primary_signal_action"] == "pass"
    assert example["hold_reference_redistributed"] is False
    assert example["local_policy_attribution_eligible"] is True
    assert [row["window_sec"] for row in example["downstream_windows"]] == [
        30, 120,
    ]
    assert example["downstream_windows"][0]["delta"]["shots"] == 1
    assert example["downstream_windows"][0]["delta"]["crosses"] == 1
    assert example["downstream_windows"][0]["schema_version"] == 2
    assert example["downstream_windows"][0][
        "causal_effect_authorized"
    ] is False

    tampered = json.loads(json.dumps(scenarios))
    tampered[0]["mechanism_examples"][0]["downstream_windows"][0][
        "delta"
    ]["goals"] = 1
    with pytest.raises(ValueError, match="window identity mismatch"):
        validate_fork_set_scenario_evidence(
            tampered, plan.branch_times_sec,
        )

    invalid_signal = json.loads(json.dumps(scenarios))
    invalid_signal[0]["mechanism_examples"][0].update({
        "signal_mode": "suppression_only",
        "recommended_action": "pass",
    })
    with pytest.raises(ValueError, match="example values"):
        validate_fork_set_scenario_evidence(
            invalid_signal, plan.branch_times_sec,
        )

    same_time = _comparison(
        plan, 1800, changed=2, local=1,
        differences=1, detailed=True,
    )
    same_time["policy_propagation"]["decisions"][1]["t_sec"] = 1810.0
    same_time["policy_propagation"]["decisions"][1]["clock"] = "30:10"
    same_time_result = aggregate_fork_set(plan, [
        same_time,
        _comparison(
            plan, 2700, changed=1, local=0,
            differences=0, detailed=True,
        ),
    ])
    same_time_scenarios = project_fork_set_scenario_evidence({
        "plan": plan.as_dict(), "rows": same_time_result["rows"],
    })
    assert len(same_time_scenarios[0]["mechanism_examples"]) == 2

    invalid_additional = _comparison(
        plan, 1800, changed=1, local=1,
        differences=1, detailed=True,
    )
    invalid_additional["policy_propagation"]["decisions"][0][
        "downstream_windows"
    ]["30s"]["additional_policy_changes"] = "unknown"
    with pytest.raises(ValueError, match="additional changes"):
        aggregate_fork_set(plan, [
            invalid_additional,
            _comparison(
                plan, 2700, changed=1, local=0,
                differences=0, detailed=True,
            ),
        ])


class _Workspace:
    def __init__(self, root, *, fail_after=None):
        self.root = root
        self.output_root = root / "outputs/studio/demo"
        self.config = SimpleNamespace(mode="research")
        self.fail_after = fail_after
        self.calls = []

    def run_paired_matches(
        self, home, away, *, fast, baseline_plan, treatment_plan, seed,
        transaction_id,
    ):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated interruption")
        branch = treatment_plan.world_model_branch_at_sec
        self.calls.append(branch)
        matches = self.output_root / "matches"
        matches.mkdir(parents=True, exist_ok=True)
        comparison = matches / f"{transaction_id}.comparison.json"
        comparison.write_text(json.dumps(
            _comparison(
                WorldModelForkSetPlan(
                    baseline_plan.home_tactic, baseline_plan.away_tactic,
                    seed, (1800, 2700, 3600),
                ),
                branch, changed=int(branch // 900), local=1, differences=1,
            )
        ), encoding="utf-8")
        comparison.with_suffix(".html").write_text("comparison", encoding="utf-8")
        return ({
            "match_id": f"base-{transaction_id}",
        }, {
            "match_id": f"treat-{transaction_id}",
            "comparison_path": str(comparison),
        })


def test_execute_fork_set_resumes_fixed_prefix_and_rejects_result_claim_tamper(tmp_path):
    plan = WorldModelForkSetPlan(
        "balanced", "low_block_counter", 42, (1800, 2700, 3600),
    )
    first = _Workspace(tmp_path, fail_after=1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute_world_model_fork_set(
            first, plan, set_id="task123", home="Brazil", away="Argentina",
            fast=True,
        )
    progress_path = first.output_root / "fork_sets/task123/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["scenarios_completed"] == 1
    assert progress["interim_ranking_disclosed"] is False
    assert progress["analysis"] is None

    resumed = _Workspace(tmp_path)
    result = execute_world_model_fork_set(
        resumed, plan, set_id="task123", home="Brazil", away="Argentina",
        fast=True,
    )
    assert resumed.calls == [2700.0, 3600.0]
    assert result["status"] == "complete"
    assert result["aggregate"]["ranking_performed"] is False
    assert result["dashboard_path"].endswith("index.html")

    result_path = first.output_root / "fork_sets/task123/result.json"
    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["claim_authority"]["match_outcome_causality"] = True
    result_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="result identity"):
        execute_world_model_fork_set(
            resumed, plan, set_id="task123", home="Brazil", away="Argentina",
            fast=True,
        )


def test_execute_fork_set_persists_manager_source_identity_in_all_artifacts(
    tmp_path,
):
    plan = WorldModelForkSetPlan(
        "gegenpress", "team_identity", 108, (1800, 2700),
    )
    season = new_season_state(
        SeasonPlan(
            ("Brazil", "B", "C", "Argentina"),
            fast=True, manager_team="Brazil",
        ),
        season_id="season-0001", seed=7,
        created_at="2026-01-01T00:00:00Z",
    )
    fixture = next(
        row for row in season["fixtures"]
        if "Brazil" in {row["home"], row["away"]}
    )
    fixture["manager_decision"] = ManagerDecision(
        team="Brazil", tactic="gegenpress",
    ).as_dict()
    source = build_manager_future_context(season)
    assert source["fixture"]["away"] == "Argentina"

    result = execute_world_model_fork_set(
        _Workspace(tmp_path), plan,
        set_id="manager123", home="Brazil", away="Argentina", fast=True,
        source_context=source,
    )

    assert result["source_context"] == source
    protocol = json.loads((
        tmp_path
        / "outputs/studio/demo/fork_sets/manager123/protocol.json"
    ).read_text(encoding="utf-8"))
    assert protocol["source_context"] == source
    dashboard = (
        tmp_path
        / "outputs/studio/demo/fork_sets/manager123/index.html"
    ).read_text(encoding="utf-8")
    assert "经理决策绑定" in dashboard
    assert "season-0001" in dashboard
    assert source["context_identity"][:16] in dashboard

    season["revision"] = 4
    changed = build_manager_future_context(season)
    with pytest.raises(ValueError, match="different frozen plan"):
        execute_world_model_fork_set(
            _Workspace(tmp_path), plan,
            set_id="manager123", home="Brazil", away="Argentina", fast=True,
            source_context=changed,
        )


def test_fork_set_task_is_idempotent_and_worker_publishes_only_bounded_claims(tmp_path):
    workspace = _Workspace(tmp_path)
    queue = ProductTaskQueue(tmp_path)
    plan = WorldModelForkSetPlan(
        "balanced", "low_block_counter", 42, (1800, 2700, 3600),
    )
    task, created = queue.submit_world_model_fork_set(
        "Brazil", "Argentina", fast=True, plan=plan,
        idempotency_key="fixed-future-set",
    )
    duplicate, duplicate_created = queue.submit_world_model_fork_set(
        "Brazil", "Argentina", fast=True, plan=plan,
        idempotency_key="fixed-future-set",
    )
    assert created is True and duplicate_created is False
    assert duplicate["task_id"] == task["task_id"]
    with pytest.raises(TaskConflict):
        queue.submit_world_model_fork_set(
            "Brazil", "Argentina", fast=True,
            plan=WorldModelForkSetPlan(
                "balanced", "low_block_counter", 42, (1200, 2400),
            ),
            idempotency_key="fixed-future-set",
        )
    assert BackgroundMatchWorker(
        queue, workspace_loader=lambda _root: workspace,
    ).run_once()
    completed = queue.get_task(task["task_id"])
    assert completed["state"] == "completed"
    assert completed["result"]["aggregate"]["ranking_performed"] is False
    assert completed["result"]["claim_authority"]["promotion_authorized"] is False
    assert completed["result"]["fork_set_dashboard"].endswith(
        "/fork_sets/" + task["task_id"] + "/index.html"
    )
    assert str(tmp_path) not in str(completed)
