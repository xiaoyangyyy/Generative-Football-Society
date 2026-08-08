import pytest

from scripts.evaluate_calibrated_m1 import build_decision
from scripts.merge_formal_ablation_results import METRICS


def _rows(delta=0.0):
    rows = []
    for fixture in ("a", "b", "c", "d", "e", "f"):
        for index in range(3):
            row = {metric: 1.0 + delta for metric in METRICS}
            row.update({"fixture": fixture, "sample_index": index})
            rows.append(row)
    return rows


def test_identical_formal_behavior_keeps_calibrated_candidate_research_only():
    baseline = {"raw_rows": _rows()}
    candidate = {
        "variant": "M1", "samples_total": 18, "all_pass": True,
        "protocol": "six_fixtures_three_seeds_full_90min_matched_to_m0",
        "raw_rows": _rows(), "loss": 1.0, "baseline_loss": 1.0,
        "delta_loss": 0.0,
        "candidate": {
            "explicit_checkpoint": True, "checkpoint": "candidate.pt",
            "checkpoint_sha256": "a" * 64,
        },
        "runtime": {"elapsed_seconds": 10.0},
    }
    result = build_decision(baseline, candidate)
    assert result["decision"] == "research_only_default_off"
    assert not result["behavior_changed_above_tolerance"]
    assert not result["promotion_supported"]


def test_formal_candidate_requires_explicit_checkpoint_provenance():
    with pytest.raises(ValueError):
        build_decision({"raw_rows": _rows()}, {
            "variant": "M1", "samples_total": 18, "raw_rows": _rows(),
            "candidate": {},
        })
