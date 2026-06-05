"""Phase 5 — additive layer gate logic (no full match sim)."""

from __future__ import annotations

from src.match_engine.calibration.contract import load_contract
from src.match_engine.calibration.layer_gate import evaluate_layer_gate, load_layer_gate_spec, metric_z_delta


def _eval_stub(*, all_pass: bool, z_map: dict[str, float]) -> dict:
    hard = {k: {"z_abs": v, "score": 0.9} for k, v in z_map.items()}
    return {
        "all_pass": all_pass,
        "hard_metrics": hard,
        "soft_constraints": {},
        "failed_hard": [],
        "failed_soft": [],
    }


def test_metric_z_delta():
    m0 = _eval_stub(all_pass=True, z_map={"pass_completion": 0.1, "goals_per_team_match": 0.2})
    m1 = _eval_stub(all_pass=False, z_map={"pass_completion": 0.15, "goals_per_team_match": 0.25})
    dz = metric_z_delta(m0, m1)
    assert abs(dz["pass_completion"] - 0.05) < 1e-9
    assert abs(dz["goals_per_team_match"] - 0.05) < 1e-9


def test_layer_gate_passes_small_delta():
    contract = load_contract()
    z = {m: 0.05 for m in load_layer_gate_spec(contract, "M1")["primary_metrics"]}
    m0 = _eval_stub(all_pass=True, z_map=z)
    m1 = _eval_stub(all_pass=False, z_map={k: v + 0.02 for k, v in z.items()})
    report = evaluate_layer_gate(m0, m1, layer="M1", contract=contract)
    assert report["all_pass"] is True
    assert report["delta_z_primary"]


def test_layer_gate_fails_large_delta():
    contract = load_contract()
    primary = load_layer_gate_spec(contract, "M1")["primary_metrics"]
    m0 = _eval_stub(all_pass=True, z_map={primary[0]: 0.1})
    m1 = _eval_stub(all_pass=False, z_map={primary[0]: 0.5})
    report = evaluate_layer_gate(m0, m1, layer="M1", contract=contract)
    assert report["all_pass"] is False
    assert primary[0] in report["violations"]


def test_layer_gate_requires_m0_pass():
    contract = load_contract()
    m0 = _eval_stub(all_pass=False, z_map={"pass_completion": 0.1})
    m1 = _eval_stub(all_pass=False, z_map={"pass_completion": 0.11})
    report = evaluate_layer_gate(m0, m1, layer="C1", contract=contract, m0_absolute_passed=False)
    assert "m0_baseline_not_passed" in report["failed"]
    report_ok = evaluate_layer_gate(m0, m1, layer="C1", contract=contract, m0_absolute_passed=True)
    assert "m0_baseline_not_passed" not in report_ok["failed"]
