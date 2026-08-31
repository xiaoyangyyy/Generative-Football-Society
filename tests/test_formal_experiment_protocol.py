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
    assert current["execution_state"] == "completed"
    assert current["completed_runs"] == {"M0": 30, "M1": 30}
    assert current["remaining_runs"] == 0
    assert current["identity_matches_progress"] is True


def test_action_outcome_result_replays_exactly():
    from scripts.verify_action_outcome_result import verify

    result = verify()
    assert result["passed"] is True
    assert result["decision"] == "inconclusive_keep_research_only"
    assert result["promotion_supported"] is False
    assert result["runs"] == 60
    assert result["pairs"] == 30
    assert result["changed_pairs"] == 30
    assert result["replay_matches"] is True


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
