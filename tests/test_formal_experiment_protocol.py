import copy
import json
import pytest

import scripts.run_formal_experiment as formal
from scripts.run_formal_experiment import (
    DEFAULT_PROTOCOL,
    _atomic_json,
    execution_identity,
    load_protocol,
    stratified_paired_interval,
    validate_protocol,
)


def test_preregistered_protocol_has_a_fixed_bounded_budget():
    protocol = load_protocol()
    assert validate_protocol(protocol) == {"fixtures": 6, "pairs": 30, "runs": 60}
    assert not protocol["integrity"]["interim_analysis"]
    assert not protocol["integrity"]["optional_stopping"]
    assert (
        protocol["analysis"]["secondary_role"] == "descriptive_only_no_promotion_claim"
    )


def test_protocol_rejects_budget_drift_and_optional_stopping():
    protocol = load_protocol()
    changed = copy.deepcopy(protocol)
    changed["design"]["runs_total"] = 62
    with pytest.raises(ValueError, match="budget"):
        validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["integrity"]["optional_stopping"] = True
    with pytest.raises(ValueError, match="optional stopping"):
        validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["design"]["arm_order"] = ["M1", "M0"]
    with pytest.raises(ValueError, match="arm order"):
        validate_protocol(changed)
    changed = copy.deepcopy(protocol)
    changed["analysis"]["bootstrap_draws"] = 9999
    with pytest.raises(ValueError, match="bootstrap budget"):
        validate_protocol(changed)


def test_execution_identity_binds_protocol_checkpoint_and_code():
    protocol = load_protocol()
    identity = execution_identity(
        DEFAULT_PROTOCOL.parents[2], DEFAULT_PROTOCOL, protocol
    )
    assert identity["checkpoint_sha256"] == protocol["candidate"]["checkpoint_sha256"]
    assert set(identity["code_sha256"]) == set(
        protocol["integrity"]["code_identity_files"]
    )
    assert set(identity["analysis_input_sha256"]) == set(
        formal.ANALYSIS_INPUT_FILES
    )


def test_atomic_progress_json_round_trips(tmp_path):
    target = tmp_path / "progress.json"
    payload = {"state": "running", "rows": [{"fixture": "A", "sample_index": 0}]}
    _atomic_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def _paired_rows(candidate_offset=0.0):
    return [
        {
            "fixture": fixture,
            "sample_index": sample,
            "synthetic_loss": float(sample) + candidate_offset,
        }
        for fixture in ("A", "B", "C")
        for sample in range(4)
    ]


def _formal_rows(protocol, *, candidate):
    rows = []
    for home, away in protocol["design"]["fixtures"]:
        for sample in protocol["design"]["sample_indices"]:
            row = {name: 1.0 for name in formal.OUTCOME_ANALYSIS_METRICS}
            row.update({
                "fixture": f"{home}_vs_{away}",
                "sample_index": sample,
                "wm_action_opportunities": 100.0 if candidate else 0.0,
                "wm_action_influenced_opportunities": (
                    100.0 if candidate else 0.0
                ),
                "wm_action_attribution_eligible_opportunities": (
                    100.0 if candidate else 0.0
                ),
                "wm_action_expected_counterfactual_changes": (
                    0.5 if candidate else 0.0
                ),
                "wm_action_counterfactual_changes": (
                    1.0 if candidate else 0.0
                ),
            })
            rows.append(row)
    return rows


def _formal_state(protocol):
    return {
        "schema_version": protocol["schema_version"],
        "protocol_id": protocol["protocol_id"],
        "state": "completed",
        "execution_identity": {"identity": "frozen"},
        "arms": {
            "M0": {
                "state": "completed",
                "rows": _formal_rows(protocol, candidate=False),
            },
            "M1": {
                "state": "completed",
                "rows": _formal_rows(protocol, candidate=True),
            },
        },
    }


def test_action_outcome_protocol_binds_confirmed_mechanism():
    path = formal.ROOT / "data/evaluation/action_outcome_protocol_v1.json"
    protocol = load_protocol(path)
    assert validate_protocol(protocol) == dict(fixtures=6, pairs=30, runs=60)
    prerequisite = protocol["prerequisite"]
    decision_path = formal.ROOT / prerequisite["mechanism_decision"]
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["status"] == prerequisite["required_status"]
    assert (
        formal.file_sha256(decision_path) == prerequisite["mechanism_decision_sha256"]
    )
    assert (
        prerequisite["mechanism_decision"]
        in protocol["integrity"]["code_identity_files"]
    )
    current = formal.status(formal.ROOT, path, protocol)
    assert current["execution_state"] == "blocked_identity_drift"
    assert current["recorded_execution_state"] == "completed"
    assert current["progress_valid"] is True
    assert all(current["progress_checks"].values())
    assert all(
        all(checks.values()) for checks in current["row_checks"].values()
    )
    assert current["completed_runs"] == {"M0": 30, "M1": 30}
    assert current["remaining_runs"] == 0
    assert current["identity_matches_progress"] is False
    assert current["next_action"] == "inspect_identity_drift"


def test_formal_progress_accepts_only_a_resumable_schedule_prefix():
    protocol = load_protocol(
        formal.ROOT / "data/evaluation/action_outcome_protocol_v1.json"
    )
    identity = {"identity": "frozen"}
    state = _formal_state(protocol)
    state["state"] = "running"
    state["arms"]["M0"].pop("state")
    state["arms"]["M0"]["rows"] = state["arms"]["M0"]["rows"][:4]
    state["arms"]["M1"].pop("state")
    state["arms"]["M1"]["rows"] = []

    report = formal.validate_progress(
        protocol, state, expected_identity=identity,
    )

    assert report["valid"] is True
    assert report["completed_runs"] == {"M0": 4, "M1": 0}
    assert report["remaining_runs"] == 56

    state["arms"]["M1"]["rows"] = _formal_rows(
        protocol, candidate=True,
    )[:1]
    rejected = formal.validate_progress(
        protocol, state, expected_identity=identity,
    )
    assert rejected["structural_valid"] is False
    assert rejected["checks"]["arm_order_is_resumable"] is False


def test_formal_status_separates_corruption_from_identity_drift(
    tmp_path, monkeypatch,
):
    path = formal.ROOT / "data/evaluation/action_outcome_protocol_v1.json"
    protocol = load_protocol(path)
    identity = {"identity": "frozen"}
    state = _formal_state(protocol)
    state["arms"]["M1"]["rows"].append(dict(
        state["arms"]["M1"]["rows"][0]
    ))
    _atomic_json(tmp_path / protocol["outputs"]["progress"], state)
    monkeypatch.setattr(formal, "execution_identity", lambda *args: identity)

    current = formal.status(tmp_path, path, protocol)

    assert current["execution_state"] == "blocked_invalid_progress"
    assert current["recorded_execution_state"] == "completed"
    assert current["progress_valid"] is False
    assert current["identity_matches_progress"] is True
    assert current["remaining_runs"] == 0
    assert current["next_action"] == "inspect_invalid_progress"


def test_formal_progress_rejects_duplicate_nonfinite_and_active_m0_rows():
    protocol = load_protocol(
        formal.ROOT / "data/evaluation/action_outcome_protocol_v1.json"
    )
    identity = {"identity": "frozen"}
    duplicate = _formal_state(protocol)
    duplicate["arms"]["M1"]["rows"][-1] = dict(
        duplicate["arms"]["M1"]["rows"][0]
    )
    with pytest.raises(ValueError, match="rows_follow_frozen_contract"):
        formal.analyze(protocol, duplicate, expected_identity=identity)

    nonfinite = _formal_state(protocol)
    nonfinite["arms"]["M1"]["rows"][0]["pass_completion"] = float("nan")
    with pytest.raises(ValueError, match="metrics_are_finite"):
        formal.analyze(protocol, nonfinite, expected_identity=identity)

    active_baseline = _formal_state(protocol)
    active_baseline["arms"]["M0"]["rows"][0][
        "wm_action_opportunities"
    ] = 1.0
    with pytest.raises(ValueError, match="baseline_has_zero_action_activity"):
        formal.analyze(protocol, active_baseline, expected_identity=identity)


def test_formal_execute_rejects_invalid_progress_before_match_runner(
    tmp_path, monkeypatch,
):
    path = formal.ROOT / "data/evaluation/action_outcome_protocol_v1.json"
    protocol = load_protocol(path)
    identity = {"identity": "frozen"}
    state = _formal_state(protocol)
    state["state"] = "running"
    state["arms"]["M0"].pop("state")
    state["arms"]["M0"]["rows"] = state["arms"]["M0"]["rows"][:2]
    state["arms"]["M1"].pop("state")
    state["arms"]["M1"]["rows"] = state["arms"]["M1"]["rows"][:1]
    _atomic_json(tmp_path / protocol["outputs"]["progress"], state)
    monkeypatch.setattr(formal, "execution_identity", lambda *args: identity)

    def forbidden_runner(**_kwargs):
        raise AssertionError("invalid progress reached the match runner")

    monkeypatch.setattr(formal, "run_micro_benchmark_rows", forbidden_runner)
    with pytest.raises(ValueError, match="arm_order_is_resumable"):
        formal.execute(
            tmp_path,
            path,
            protocol,
            authorization=formal.AUTHORIZATION,
        )


def test_formal_execute_is_idempotent_for_a_complete_valid_ledger(
    tmp_path, monkeypatch,
):
    path = formal.ROOT / "data/evaluation/action_outcome_protocol_v1.json"
    protocol = load_protocol(path)
    identity = {"identity": "frozen"}
    state = _formal_state(protocol)
    progress_path = tmp_path / protocol["outputs"]["progress"]
    _atomic_json(progress_path, state)
    monkeypatch.setattr(formal, "execution_identity", lambda *args: identity)

    def forbidden_runner(**_kwargs):
        raise AssertionError("complete progress reached the match runner")

    monkeypatch.setattr(formal, "run_micro_benchmark_rows", forbidden_runner)
    result = formal.execute(
        tmp_path,
        path,
        protocol,
        authorization=formal.AUTHORIZATION,
    )

    assert result == state
    assert json.loads(progress_path.read_text(encoding="utf-8")) == state


def test_formal_execute_rejects_incomplete_m0_before_starting_m1(
    tmp_path, monkeypatch,
):
    path = formal.ROOT / "data/evaluation/action_outcome_protocol_v1.json"
    protocol = load_protocol(path)
    identity = {"identity": "frozen"}
    monkeypatch.setattr(formal, "execution_identity", lambda *args: identity)
    calls = []

    def incomplete_runner(**kwargs):
        calls.append(kwargs["spec"].name)
        return []

    monkeypatch.setattr(formal, "run_micro_benchmark_rows", incomplete_runner)
    with pytest.raises(ValueError, match="arm_completion_is_consistent"):
        formal.execute(
            tmp_path,
            path,
            protocol,
            authorization=formal.AUTHORIZATION,
        )

    assert calls == ["M0"]
    progress = json.loads(
        (tmp_path / protocol["outputs"]["progress"]).read_text(encoding="utf-8")
    )
    assert progress["state"] == "failed"
    assert progress["arms"]["M1"]["rows"] == []


def test_formal_execution_requires_exact_explicit_authorization(tmp_path):
    path = formal.ROOT / "data/evaluation/action_outcome_protocol_v1.json"
    protocol = load_protocol(path)

    with pytest.raises(PermissionError, match="explicit formal experiment"):
        formal.execute(tmp_path, path, protocol)
    with pytest.raises(PermissionError, match="explicit formal experiment"):
        formal.execute(tmp_path, path, protocol, authorization="yes")
    assert not (tmp_path / protocol["outputs"]["progress"]).exists()


def test_historical_action_outcome_result_rejects_changed_controller_identity():
    from scripts.verify_action_outcome_result import verify

    with pytest.raises(ValueError, match="identity"):
        verify()


def test_fixture_stratified_bootstrap_preserves_exact_pairing(monkeypatch):
    monkeypatch.setattr(
        formal,
        "_loss",
        lambda rows: (
            sum(row["synthetic_loss"] for row in rows) / len(rows),
            {},
        ),
    )
    interval = stratified_paired_interval(
        _paired_rows(),
        _paired_rows(0.25),
        draws=50,
        seed=7,
    )
    assert interval["point_delta"] == pytest.approx(0.25)
    assert interval["ci95_low"] == pytest.approx(0.25)
    assert interval["ci95_high"] == pytest.approx(0.25)


def test_fixture_stratified_bootstrap_rejects_unpaired_units():
    with pytest.raises(ValueError, match="do not match exactly"):
        stratified_paired_interval(
            _paired_rows(),
            _paired_rows()[:-1],
            draws=2,
            seed=7,
        )
