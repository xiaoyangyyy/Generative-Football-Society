import copy
import json
from pathlib import Path

import pytest

from src.product.excellence import derive_excellence, validate_scoring_contract

CONTRACT = Path("data/evaluation/excellence_scoring_contract_v1.json")


def _contract():
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_current_missing_external_evidence_derives_83_and_70():
    result = derive_excellence(
        _contract(),
        {},
        baseline_scores={"product": 58, "academic": 62},
    )
    assert result["tracks"]["product"]["score"] == 83
    assert result["tracks"]["academic"]["score"] == 70
    assert result["all_tracks_full_maturity"] is False


def test_all_required_evidence_gates_derive_exactly_100_per_track():
    contract = _contract()
    gates = {
        gate: True
        for rows in contract["tracks"].values()
        for row in rows
        for gate in row["completion_requires"]
    }
    result = derive_excellence(contract, gates)
    assert result["tracks"]["product"]["score"] == 100
    assert result["tracks"]["academic"]["score"] == 100
    assert result["all_tracks_full_maturity"] is True
    assert all(
        category["status"] == "verified"
        for track in result["tracks"].values()
        for category in track["categories"]
    )


def test_product_full_score_requires_outcomes_and_security_closure():
    contract = _contract()
    gates = {
        gate: True
        for row in contract["tracks"]["product"]
        for gate in row["completion_requires"]
    }
    gates["credential_security_closure"] = False
    result = derive_excellence(contract, gates)
    product = result["tracks"]["product"]
    assert product["score"] == 99
    p2 = next(row for row in product["categories"] if row["id"] == "P2")
    assert p2["missing_gates"] == ["credential_security_closure"]


def test_valid_completed_negative_research_can_reach_academic_100():
    contract = _contract()
    gates = {
        "confirmatory_results": True,
        "independent_reproduction": True,
        "completed_manuscript": True,
    }
    result = derive_excellence(contract, gates)
    assert result["tracks"]["academic"]["score"] == 100


def test_contract_tampering_fails_closed():
    contract = _contract()
    assert all(validate_scoring_contract(contract).values())
    tampered = copy.deepcopy(contract)
    tampered["tracks"]["product"][0]["weight"] = 19
    assert not validate_scoring_contract(tampered)["each_track_totals_exactly_100"]
    with pytest.raises(ValueError, match="invalid excellence scoring contract"):
        derive_excellence(tampered, {})
