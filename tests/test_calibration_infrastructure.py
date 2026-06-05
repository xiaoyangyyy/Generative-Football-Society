"""Fast tests for calibration infrastructure (no full match sim)."""

from __future__ import annotations

import json
from pathlib import Path

from src.match_engine.calibration.ablation import ADDITIVE_LAYERS, ABLATION_PRESETS, PIPELINE_PRESETS, apply_ablation
from src.match_engine.calibration.contract import evaluate_rows, load_contract
from src.match_engine.calibration.profile import load_profile
from src.match_engine.pass_calibration import logit_target_blend


ROOT = Path(__file__).resolve().parents[1]


def test_observable_contract_loads():
    c = load_contract()
    assert c["version"] == "1"
    assert "M0" in c["pipelines"]
    assert "C1" in c["pipelines"]
    assert "M1" in c.get("layer_gates", {})
    assert len(c["hard_metrics"]["metrics"]) >= 10


def test_param_registry_profile():
    p = load_profile()
    assert p.name == "v5_physics_first"
    assert len(p.parameters) >= 15
    assert abs(p.get("pass.logit_target_blend") - 0.7) < 1e-9


def test_evaluate_rows_hard_and_soft():
    rows = [
        {
            "fixture": "A_vs_B",
            "pass_completion": 0.81,
            "interceptions_per_pass": 0.02,
            "passes_per_team_match": 500.0,
            "through_share": 0.003,
            "long_pass_share": 0.19,
            "shots_per_team_match": 11.0,
            "shots_on_target_rate": 0.35,
            "goals_per_team_match": 1.5,
            "fouls_committed_per_team_match": 14.0,
            "yellow_cards_per_team_match": 0.3,
            "red_cards_per_team_match": 0.0,
            "possession_share": 0.52,
            "crosses_per_team_match": 11.0,
            "headers_per_team_match": 2.0,
            "tackles_per_team_match": 17.0,
            "micro_xg_per_team_match": 1.2,
            "goals_to_micro_xg_ratio": 1.1,
        },
        {
            "fixture": "C_vs_D",
            "pass_completion": 0.80,
            "interceptions_per_pass": 0.021,
            "passes_per_team_match": 520.0,
            "through_share": 0.004,
            "long_pass_share": 0.20,
            "shots_per_team_match": 12.0,
            "shots_on_target_rate": 0.33,
            "goals_per_team_match": 1.2,
            "fouls_committed_per_team_match": 13.0,
            "yellow_cards_per_team_match": 0.4,
            "red_cards_per_team_match": 0.0,
            "possession_share": 0.48,
            "crosses_per_team_match": 10.0,
            "headers_per_team_match": 1.5,
            "tackles_per_team_match": 16.0,
            "micro_xg_per_team_match": 1.1,
            "goals_to_micro_xg_ratio": 0.95,
        },
    ]
    ev = evaluate_rows(rows)
    assert ev["samples_total"] == 2
    assert "pass_completion" in ev["hard_metrics"]


def test_ablation_presets_exist():
    assert "M0" in PIPELINE_PRESETS
    assert "no_blend" in ABLATION_PRESETS
    assert "M1" in ADDITIVE_LAYERS
    cfg = apply_ablation(ABLATION_PRESETS["no_affective"])
    assert cfg.disable_affective_dynamics is True


def test_no_blend_env():
    import os

    old = os.environ.get("CALIBRATION_NO_BLEND")
    os.environ["CALIBRATION_NO_BLEND"] = "1"
    try:
        assert logit_target_blend(base_dir=str(ROOT)) == 0.0
    finally:
        if old is None:
            os.environ.pop("CALIBRATION_NO_BLEND", None)
        else:
            os.environ["CALIBRATION_NO_BLEND"] = old


def test_registry_json_valid():
    data = json.loads((ROOT / "data" / "calibration" / "param_registry.json").read_text(encoding="utf-8"))
    ids = [p["id"] for p in data["parameters"]]
    assert len(ids) == len(set(ids))
    assert "discipline.cross_tick_target" in ids


def test_aerial_header_stats_and_xg_helper():
    from src.match_engine.aerial_duel import AerialDuelEngine, apply_aerial_xg_to_state
    from src.match_engine.micro_config import MicroMatchConfig
    from src.match_engine.state import CrowdState, MatchAffectiveState, RefereeAffectiveState, TeamAffectiveState

    cfg = MicroMatchConfig()
    aerial = AerialDuelEngine(cfg)
    aerial.stats["headers"] = 3
    aerial.stats["home_headers"] = 2
    aerial.stats["away_headers"] = 1
    shot_headers = 1
    headers_total = aerial.stats["headers"] + shot_headers
    assert headers_total == 4

    state = MatchAffectiveState(
        home=TeamAffectiveState(team_id="H"),
        away=TeamAffectiveState(team_id="A"),
        referee=RefereeAffectiveState(),
        crowd=CrowdState(),
    )
    apply_aerial_xg_to_state(state, 0.25, True, cfg)
    assert state.micro_xg_home == 0.25
