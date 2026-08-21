import copy
import json
from pathlib import Path

import pytest

from src.product.comparison import build_paired_comparison
from src.product.tactical_study import (
    TacticalStudyPlan, analyze_fixed_tactical_study,
    execute_tactical_study, render_tactical_study_html,
)
from tests.test_product_comparison import _report
from tests.test_product_workspace import _Summary, _evidence
from src.product.workspace import ProductWorkspace, StudioConfig


def _plan(seeds=(11, 12, 13, 14)):
    return TacticalStudyPlan(
        study_id="brazil-press-v1", home="Brazil", away="Argentina",
        baseline_home_tactic="gegenpress",
        baseline_away_tactic="low_block_counter",
        treatment_home_tactic="counter_attack",
        treatment_away_tactic="low_block_counter",
        seeds=seeds,
    )


def _comparison(seed, xg_delta):
    baseline = _report(f"b-{seed}")
    treatment = _report(
        f"t-{seed}", tactic="counter_attack", reuse=True,
        baseline=f"b-{seed}",
    )
    baseline["fixture"]["seed"] = treatment["fixture"]["seed"] = seed
    treatment["result"]["xg"]["home"] += xg_delta
    return build_paired_comparison(baseline, treatment)


def test_plan_freezes_single_side_budget_metrics_and_claim_boundary():
    plan = _plan()
    payload = plan.as_dict()
    assert plan.focus_side == "home"
    assert payload["fixed_pair_budget"] == 4
    assert payload["analysis_plan"]["primary_metric"] == "focus_xg_difference"
    assert payload["analysis_plan"]["interim_analysis"] == (
        "withheld_until_fixed_budget_complete"
    )
    assert "not a cross-fixture population effect" in payload["claim_boundary"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": (1, 2, 3)},
        {"seeds": (1, 1, 2, 3)},
        {"treatment_away_tactic": "balanced"},
    ],
)
def test_invalid_budget_duplicate_seed_or_joint_intervention_is_rejected(kwargs):
    values = dict(
        study_id="study-one", home="Brazil", away="Argentina",
        baseline_home_tactic="gegenpress",
        baseline_away_tactic="low_block_counter",
        treatment_home_tactic="counter_attack",
        treatment_away_tactic="low_block_counter",
        seeds=(1, 2, 3, 4),
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        TacticalStudyPlan(**values)


def test_incomplete_budget_withholds_all_effects_even_when_rows_exist():
    result = analyze_fixed_tactical_study(
        _plan(), [_comparison(11, 0.2), _comparison(12, 0.3)],
    )
    assert result["status"] == "incomplete_analysis_withheld"
    assert result["analysis"] is None
    assert not result["interim_effects_disclosed"]
    assert "rows" not in result
    document = render_tactical_study_html(result)
    assert "禁止披露中期效果" in document
    assert "0.2" not in document and "0.3" not in document


def test_complete_fixed_budget_reports_t_interval_sign_test_and_no_promotion():
    result = analyze_fixed_tactical_study(
        _plan(), [
            _comparison(11, 0.20), _comparison(12, 0.24),
            _comparison(13, 0.18), _comparison(14, 0.22),
        ],
    )
    primary = result["analysis"]["summaries"]["focus_xg_difference"]
    assert result["status"] == "complete"
    assert primary["mean"] == pytest.approx(0.21)
    assert primary["ci_low"] > 0
    assert primary["sign_test"]["two_sided_exact_p"] == pytest.approx(0.125)
    assert result["analysis"]["primary_decision"] == (
        "beneficial_on_fixed_seed_set"
    )
    assert not result["analysis"]["promotion_authorized"]
    assert result["analysis"]["secondary_metrics_are_descriptive"]
    assert all(row["primary_direction"] == "positive" for row in result["rows"])
    assert all(row["supports_aggregate_direction"] for row in result["rows"])


def test_final_dashboard_links_only_safe_match_evidence():
    result = analyze_fixed_tactical_study(
        _plan(), [_comparison(seed, 0.2) for seed in _plan().seeds],
    )
    result["rows"][0]["evidence"] = {
        "baseline_dashboard": "outputs/studio/demo/matches/0001-base.html",
        "treatment_dashboard": "outputs/studio/demo/matches/0002-treatment.html",
        "comparison_dashboard": (
            "outputs/studio/demo/matches/0002-paired.comparison.html"
        ),
    }
    result["rows"][1]["evidence"] = {
        "baseline_dashboard": "javascript:alert(1)",
        "comparison_dashboard": "outputs/studio/../../secret.html",
    }
    document = render_tactical_study_html(result)
    assert "固定 seed 证据下钻" in document
    assert '../../matches/0001-base.html' in document
    assert '../../matches/0002-treatment.html' in document
    assert '../../matches/0002-paired.comparison.html' in document
    assert "javascript:" not in document
    assert "../../secret" not in document
    assert "证据链接不可用" in document


def test_ineligible_tampered_or_missing_pair_fails_instead_of_complete_case_drop():
    plan = _plan()
    rows = [_comparison(seed, 0.2) for seed in plan.seeds]
    ineligible = copy.deepcopy(rows)
    ineligible[0]["eligibility"]["eligible_for_tactical_attribution"] = False
    with pytest.raises(ValueError, match="ineligible"):
        analyze_fixed_tactical_study(plan, ineligible)
    missing = copy.deepcopy(rows)
    missing[0]["metrics"]["xg_home"]["delta"] = None
    with pytest.raises(ValueError, match="missing comparison metric"):
        analyze_fixed_tactical_study(plan, missing)


def test_duplicate_or_out_of_plan_seed_is_rejected():
    plan = _plan()
    row = _comparison(11, 0.2)
    with pytest.raises(ValueError, match="duplicate"):
        analyze_fixed_tactical_study(plan, [row, row])
    outside = _comparison(99, 0.2)
    with pytest.raises(ValueError, match="outside frozen"):
        analyze_fixed_tactical_study(plan, [outside])


def test_plan_round_trip_preserves_frozen_identity():
    plan = _plan()
    assert TacticalStudyPlan.from_payload(plan.as_dict()).as_dict() == plan.as_dict()


def test_executor_runs_fixed_pairs_once_withholds_progress_and_is_idempotent(
    tmp_path, monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Study Lab", mode="research", seed=3),
    )
    calls = []
    from src import app

    def fake_match(*args, **kwargs):
        calls.append(kwargs)
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", fake_match)
    result = execute_tactical_study(workspace, _plan())
    assert result["status"] == "complete"
    assert len(calls) == 8
    assert Path(result["result_path"]).is_file()
    assert Path(result["dashboard_path"]).is_file()
    progress_path = (
        workspace.output_root / "studies/brazil-press-v1/progress.json"
    )
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["pairs_completed"] == 4
    assert progress["analysis"] is None
    assert not progress["interim_effects_disclosed"]
    assert all("metrics" not in row for row in progress["completed"])
    assert all("evidence" not in row for row in progress["completed"])
    assert all(set(row["evidence"]) == {
        "baseline_dashboard", "treatment_dashboard", "comparison_dashboard",
    } for row in result["rows"])
    dashboard = Path(result["dashboard_path"]).read_text(encoding="utf-8")
    assert dashboard.count("../../matches/") == 12
    assert "基线" in dashboard and "处理" in dashboard and "配对" in dashboard
    Path(result["dashboard_path"]).unlink()
    repeated = execute_tactical_study(workspace, _plan())
    assert repeated["status"] == "complete"
    assert len(calls) == 8
    assert Path(repeated["dashboard_path"]).is_file()


def test_executor_rejects_same_study_id_with_changed_frozen_plan(
    tmp_path, monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Study Lab", mode="research", seed=3),
    )
    from src import app
    monkeypatch.setattr(app, "run_micro_match", lambda *args, **kwargs: _Summary())
    execute_tactical_study(workspace, _plan())
    changed = TacticalStudyPlan(
        study_id="brazil-press-v1", home="Brazil", away="Argentina",
        baseline_home_tactic="gegenpress",
        baseline_away_tactic="low_block_counter",
        treatment_home_tactic="tiki_taka",
        treatment_away_tactic="low_block_counter",
        seeds=(11, 12, 13, 14),
    )
    with pytest.raises(ValueError, match="different frozen plan"):
        execute_tactical_study(workspace, changed)


def test_executor_rejects_progress_redirect_to_non_match_json(
    tmp_path, monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Study Lab", mode="research", seed=3),
    )
    from src import app
    calls = []

    def fake_match(*args, **kwargs):
        calls.append(kwargs)
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", fake_match)
    result = execute_tactical_study(workspace, _plan())
    result_path = Path(result["result_path"])
    result_path.unlink()
    Path(result["dashboard_path"]).unlink()
    private = tmp_path / "data/private.json"
    private.write_text('{"secret": true}', encoding="utf-8")
    progress_path = workspace.output_root / "studies/brazil-press-v1/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["completed"][0]["comparison"] = "data/private.json"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    with pytest.raises(ValueError, match="inside Studio matches"):
        execute_tactical_study(workspace, _plan())
    failed = json.loads(progress_path.read_text(encoding="utf-8"))
    assert failed["state"] == "failed"
    assert failed["analysis"] is None
    assert not failed["interim_effects_disclosed"]
    assert len(calls) == 8


def test_executor_revalidates_completed_result_before_republishing(
    tmp_path, monkeypatch,
):
    _evidence(tmp_path)
    workspace = ProductWorkspace.create(
        tmp_path, StudioConfig(name="Study Lab", mode="research", seed=3),
    )
    from src import app
    calls = []

    def fake_match(*args, **kwargs):
        calls.append(kwargs)
        return _Summary()

    monkeypatch.setattr(app, "run_micro_match", fake_match)
    completed = execute_tactical_study(workspace, _plan())
    result_path = Path(completed["result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["analysis"]["promotion_authorized"] = True
    result_path.write_text(json.dumps(result), encoding="utf-8")
    Path(completed["dashboard_path"]).unlink()

    with pytest.raises(ValueError, match="result identity is invalid"):
        execute_tactical_study(workspace, _plan())
    assert not Path(completed["dashboard_path"]).exists()
    assert len(calls) == 8
