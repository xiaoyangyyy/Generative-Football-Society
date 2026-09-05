"""Preregistered action-adoption mechanism study is bounded and fail-closed."""

from __future__ import annotations

import pytest

from scripts import action_adoption_study as study
from scripts.merge_formal_ablation_results import METRICS


def _rows(protocol, *, candidate: bool, realized_changes: float = 1.0):
    rows = []
    for home, away in protocol["design"]["fixtures"]:
        for sample in protocol["design"]["sample_indices"]:
            row = {metric: 1.0 for metric in METRICS}
            row.update({
                "fixture": f"{home}_vs_{away}",
                "sample_index": sample,
                "wm_action_opportunities": 100.0 if candidate else 0.0,
                "wm_action_influenced_opportunities": 100.0 if candidate else 0.0,
                "wm_action_attribution_eligible_opportunities": (
                    100.0 if candidate else 0.0
                ),
                "wm_action_expected_counterfactual_changes": (
                    0.5 if candidate else 0.0
                ),
                "wm_action_counterfactual_changes": (
                    realized_changes if candidate else 0.0
                ),
            })
            if candidate:
                row[METRICS[0]] += 0.01
            rows.append(row)
    return rows


def _state(protocol, *, realized_changes: float = 1.0):
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "state": "completed",
        "execution_identity": {"identity": "frozen"},
        "material_deviations": [],
        "arms": {
            "M1_predict_only": {
                "state": "completed",
                "rows": _rows(protocol, candidate=False),
            },
            "M1_action_policy": {
                "state": "completed",
                "rows": _rows(
                    protocol, candidate=True,
                    realized_changes=realized_changes,
                )
            },
        },
    }


def _legacy_preexecution_assertions():
    report = study.protocol_report()
    current = study.status()

    assert report["passed"] is True
    assert report["ready_to_start"] is True
    assert all(report["checks"].values())
    assert report["runs_executed"] == 0
    assert report["training_executed"] is False
    assert report["provider_calls_made"] is False
    assert current["state"] == "ready_not_started"
    assert current["remaining_runs"] == 24
    assert current["identity_matches_progress"] is True


def test_completed_mechanism_result_is_blocked_after_controller_identity_drift():
    from scripts.verify_action_adoption_result import verify

    report = study.protocol_report()
    current = study.status()
    assert report["passed"] is True
    assert report["ready_to_start"] is True
    assert all(report["checks"].values())
    assert current["state"] == "blocked_identity_drift"
    assert current["progress_valid"] is True
    assert all(current["progress_checks"].values())
    assert all(
        all(checks.values()) for checks in current["row_checks"].values()
    )
    assert current["completed_runs"] == dict(
        M1_predict_only=12,
        M1_action_policy=12,
    )
    assert current["remaining_runs"] == 0
    assert current["identity_matches_progress"] is False
    assert current["next_action"] == "inspect_identity_drift"
    with pytest.raises(ValueError, match="identity"):
        verify()


def test_progress_validator_accepts_only_a_resumable_schedule_prefix():
    protocol = study._read_json(study.PROTOCOL_PATH)
    identity = {"identity": "frozen"}
    state = _state(protocol)
    state["state"] = "running"
    state["arms"]["M1_predict_only"].pop("state")
    state["arms"]["M1_predict_only"]["rows"] = state["arms"][
        "M1_predict_only"
    ]["rows"][:3]
    state["arms"]["M1_action_policy"].pop("state")
    state["arms"]["M1_action_policy"]["rows"] = []

    report = study.validate_progress(
        protocol, state, expected_identity=identity,
    )

    assert report["valid"] is True
    assert report["completed_runs"] == {
        "M1_predict_only": 3,
        "M1_action_policy": 0,
    }
    assert report["remaining_runs"] == 21

    state["arms"]["M1_action_policy"]["rows"] = _rows(
        protocol, candidate=True,
    )[:1]
    rejected = study.validate_progress(
        protocol, state, expected_identity=identity,
    )
    assert rejected["structural_valid"] is False
    assert rejected["checks"]["arm_order_is_resumable"] is False


def test_status_distinguishes_invalid_progress_from_identity_drift(
    tmp_path, monkeypatch,
):
    protocol = study._read_json(study.PROTOCOL_PATH)
    identity = {"identity": "frozen"}
    state = _state(protocol)
    state["arms"]["M1_action_policy"]["rows"].append(dict(
        state["arms"]["M1_action_policy"]["rows"][0]
    ))
    progress_path = tmp_path / protocol["outputs"]["progress"]
    study._atomic_json(progress_path, state)
    monkeypatch.setattr(study, "execution_identity", lambda *args, **kwargs: identity)

    current = study.status(root=tmp_path)

    assert current["state"] == "blocked_invalid_progress"
    assert current["progress_valid"] is False
    assert current["identity_matches_progress"] is True
    assert current["next_action"] == "inspect_invalid_progress"
    assert current["remaining_runs"] == 0
    assert current["progress_checks"][
        "recorded_rows_do_not_exceed_budget"
    ] is False


def test_analysis_rejects_duplicate_or_nonfinite_fixed_schedule_rows():
    protocol = study._read_json(study.PROTOCOL_PATH)
    identity = {"identity": "frozen"}
    duplicate = _state(protocol)
    duplicate["arms"]["M1_action_policy"]["rows"][-1] = dict(
        duplicate["arms"]["M1_action_policy"]["rows"][0]
    )
    with pytest.raises(ValueError, match="rows_follow_frozen_contract"):
        study.analyze(protocol, duplicate, expected_identity=identity)

    nonfinite = _state(protocol)
    nonfinite["arms"]["M1_action_policy"]["rows"][0][
        "wm_action_opportunities"
    ] = float("nan")
    with pytest.raises(ValueError, match="analysis_metrics_are_finite"):
        study.analyze(protocol, nonfinite, expected_identity=identity)

    impossible_counts = _state(protocol)
    impossible_counts["arms"]["M1_action_policy"]["rows"][0][
        "wm_action_attribution_eligible_opportunities"
    ] = 101.0
    with pytest.raises(ValueError, match="action_counts_are_bounded"):
        study.analyze(protocol, impossible_counts, expected_identity=identity)


def test_execute_rejects_invalid_progress_before_match_runner(
    tmp_path, monkeypatch,
):
    protocol = study._read_json(study.PROTOCOL_PATH)
    identity = {"identity": "frozen"}
    state = _state(protocol)
    state["state"] = "running"
    state["arms"]["M1_predict_only"].pop("state")
    state["arms"]["M1_predict_only"]["rows"] = state["arms"][
        "M1_predict_only"
    ]["rows"][:2]
    state["arms"]["M1_action_policy"].pop("state")
    state["arms"]["M1_action_policy"]["rows"] = state["arms"][
        "M1_action_policy"
    ]["rows"][:1]
    progress_path = tmp_path / protocol["outputs"]["progress"]
    study._atomic_json(progress_path, state)
    monkeypatch.setattr(study, "execution_identity", lambda *args, **kwargs: identity)

    def forbidden_runner(**_kwargs):
        raise AssertionError("invalid progress reached the match runner")

    monkeypatch.setattr(study, "run_micro_benchmark_rows", forbidden_runner)
    with pytest.raises(ValueError, match="arm_order_is_resumable"):
        study.execute(authorization=study.AUTHORIZATION, root=tmp_path)


def test_execute_is_idempotent_for_a_complete_identity_matched_ledger(
    tmp_path, monkeypatch,
):
    protocol = study._read_json(study.PROTOCOL_PATH)
    identity = {"identity": "frozen"}
    state = _state(protocol)
    progress_path = tmp_path / protocol["outputs"]["progress"]
    study._atomic_json(progress_path, state)
    monkeypatch.setattr(study, "execution_identity", lambda *args, **kwargs: identity)

    def forbidden_runner(**_kwargs):
        raise AssertionError("complete progress reached the match runner")

    monkeypatch.setattr(study, "run_micro_benchmark_rows", forbidden_runner)
    result = study.execute(authorization=study.AUTHORIZATION, root=tmp_path)

    assert result == state
    assert study._read_json(progress_path) == state


def test_execute_rejects_an_incomplete_arm_before_starting_the_next(
    tmp_path, monkeypatch,
):
    protocol = study._read_json(study.PROTOCOL_PATH)
    identity = {"identity": "frozen"}
    monkeypatch.setattr(study, "execution_identity", lambda *args, **kwargs: identity)
    calls = []

    def incomplete_runner(**kwargs):
        calls.append(kwargs["spec"].name)
        return []

    monkeypatch.setattr(study, "run_micro_benchmark_rows", incomplete_runner)
    with pytest.raises(ValueError, match="arm_completion_is_consistent"):
        study.execute(authorization=study.AUTHORIZATION, root=tmp_path)

    assert calls == ["M1"]
    progress_path = tmp_path / protocol["outputs"]["progress"]
    progress = study._read_json(progress_path)
    assert progress["state"] == "failed"
    assert progress["arms"]["M1_action_policy"]["rows"] == []


def test_execution_requires_exact_explicit_authorization():
    with pytest.raises(PermissionError, match="explicit action-adoption"):
        study.execute(authorization="yes")


def test_analysis_confirms_only_mechanism_and_never_promotes():
    protocol = study._read_json(study.PROTOCOL_PATH)
    identity = {"identity": "frozen"}

    result = study.analyze(
        protocol, _state(protocol), expected_identity=identity,
    )

    assert result["passed"] is True
    assert result["status"] == "mechanism_confirmed"
    assert result["claim_scope"] == (
        "mechanism_confirmation_only_no_product_or_academic_promotion"
    )
    assert result["promotion_authorized"] is False
    assert result["runs_executed"] == 24
    assert all(result["gates"].values())


def test_analysis_fails_closed_when_realized_action_change_gate_misses():
    protocol = study._read_json(study.PROTOCOL_PATH)
    result = study.analyze(
        protocol,
        _state(protocol, realized_changes=0.0),
        expected_identity={"identity": "frozen"},
    )

    assert result["passed"] is False
    assert result["status"] == "mechanism_not_confirmed"
    assert result["gates"]["minimum_realized_counterfactual_changes"] is False
    assert result["promotion_authorized"] is False


def test_four_arm_smoke_is_diagnostic_and_does_not_touch_formal_progress(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        study, "execution_identity",
        lambda *args, **kwargs: {"identity": "smoke"},
    )
    calls = []

    def runner(**kwargs):
        calls.append(kwargs["spec"].name)
        planning = study.os.environ.get("MATCH_WM_PLAN") == "1"
        return [{
            "fixture": "Brazil_vs_Germany",
            "sample_index": 0,
            "passes_per_team_match": 10.0,
            "shots_per_team_match": 1.0,
            "wm_action_opportunities": (
                5.0 if planning else 0.0
            ),
            "wm_action_influenced_opportunities": 5.0 if planning else 0.0,
            "wm_action_attribution_eligible_opportunities": (
                5.0 if planning else 0.0
            ),
        }]

    output = tmp_path / "smoke.json"
    result = study.smoke(root=tmp_path, output_path=output, runner=runner)

    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["runs_executed"] == 4
    assert list(result["arms"]) == list(study.SMOKE_ARM_IDS)
    assert calls == ["M0", "C1", "M1", "M1"]
    assert result["diagnostic_only"] is True
    assert result["formal_progress_modified"] is False
    assert result["training_executed"] is False
    assert result["provider_calls_made"] is False
    assert output.is_file()
    assert not (
        tmp_path / "data/evaluation/action_adoption_v1/progress.json"
    ).exists()


def test_completed_analysis_exports_flat_rows_and_bounded_summary(tmp_path):
    protocol = study._read_json(study.PROTOCOL_PATH)
    identity = {"identity": "frozen"}
    state = _state(protocol)
    result = study.analyze(protocol, state, expected_identity=identity)

    exported = study.export_analysis_artifacts(
        protocol, state, result, root=tmp_path,
    )

    csv_path = tmp_path / protocol["outputs"]["rows_csv"]
    summary_path = tmp_path / protocol["outputs"]["summary_markdown"]
    assert exported["rows_csv"]["rows"] == 24
    assert csv_path.read_text(encoding="utf-8").count("\n") == 25
    summary = summary_path.read_text(encoding="utf-8")
    assert "mechanism_confirmed" in summary
    assert "Training executed: no" in summary
    assert "does not authorize product or academic promotion" in summary
