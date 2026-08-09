import copy
import json
from pathlib import Path

from scripts.verify_excellence_score import verify_excellence_score
from src.product.excellence import derive_excellence

CONTRACT = Path("data/evaluation/excellence_scoring_contract_v1.json")
STORED_REPORT = Path("data/evaluation/excellence_score_verification_v1.json")


def test_dynamic_score_report_matches_current_control_plane():
    report = verify_excellence_score()
    assert report["passed"] is True
    assert report["status"] == "passed_dynamic_scoring"
    assert report["scores"] == {"product": 83, "academic": 70}
    assert report["all_tracks_full_maturity"] is False
    assert report["release_ready"] is False
    assert report["passed_gate_count"] == 11
    assert report["open_gate_count"] == 10
    assert all(report["checks"].values())
    assert report["checks"][
        "completion_plan_is_dependency_aware_and_zero_execution"
    ] is True
    assert "src/product/completion_plan.py" in report["artifact_sha256"]
    assert "src/product/web.py" in report["artifact_sha256"]
    assert "scripts/build_excellence_evidence_kit.py" in report["artifact_sha256"]
    assert report["matches_executed"] == 0


def test_stored_dynamic_score_snapshot_is_current():
    stored = json.loads(STORED_REPORT.read_text(encoding="utf-8"))
    current = verify_excellence_score()
    assert stored["passed"] is True
    assert stored["scores"] == current["scores"]
    assert stored["checks"] == current["checks"]
    assert stored["artifact_sha256"] == current["artifact_sha256"]
    assert stored["passed_gate_count"] == current["passed_gate_count"]
    assert stored["open_gate_count"] == current["open_gate_count"]


def test_favorable_direction_cannot_replace_missing_academic_evidence():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    gates = {"confirmatory_results": True}
    result = derive_excellence(contract, gates)
    academic = result["tracks"]["academic"]
    assert academic["score"] == 85
    a6 = next(row for row in academic["categories"] if row["id"] == "A6")
    assert a6["missing_gates"] == ["independent_reproduction"]
    a7 = next(row for row in academic["categories"] if row["id"] == "A7")
    assert set(a7["missing_gates"]) == {
        "independent_reproduction",
        "completed_manuscript",
    }


def test_manual_roadmap_points_do_not_change_dynamic_score():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    tampered_roadmap = copy.deepcopy(contract)
    tampered_roadmap["untrusted_current_points"] = 100
    result = derive_excellence(tampered_roadmap, {})
    assert result["tracks"]["product"]["score"] == 83
    assert result["tracks"]["academic"]["score"] == 70
