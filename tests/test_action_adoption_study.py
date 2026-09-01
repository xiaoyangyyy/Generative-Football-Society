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
            "M1_predict_only": {"rows": _rows(protocol, candidate=False)},
            "M1_action_policy": {
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
    assert report['passed'] is True
    assert report['ready_to_start'] is True
    assert all(report['checks'].values())
    assert current['state'] == 'blocked_identity_drift'
    assert current['completed_runs'] == dict(
        M1_predict_only=12,
        M1_action_policy=12,
    )
    assert current['remaining_runs'] == 0
    assert current['identity_matches_progress'] is False
    assert current['next_action'] == 'inspect_identity_drift'
    with pytest.raises(ValueError, match='identity'):
        verify()
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
