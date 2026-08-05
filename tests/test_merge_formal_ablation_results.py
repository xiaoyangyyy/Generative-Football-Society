from scripts.merge_formal_ablation_results import _decision, paired_intervals


def _rows(offset=0.0):
    rows = []
    for fixture in ("A", "B"):
        for seed in range(3):
            row = {key: 0.5 + seed * 0.01 + offset for key in (
                "pass_completion", "interceptions_per_pass", "passes_per_team_match",
                "long_pass_share", "shots_per_team_match", "shots_on_target_rate",
                "goals_per_team_match", "fouls_committed_per_team_match",
                "yellow_cards_per_team_match", "red_cards_per_team_match",
                "possession_share", "crosses_per_team_match", "headers_per_team_match",
                "tackles_per_team_match", "micro_xg_per_team_match",
            )}
            row["fixture"] = fixture
            rows.append(row)
    return rows


def test_paired_metric_intervals_preserve_constant_delta():
    result = paired_intervals(_rows(), _rows(0.25), draws=20)
    assert result["metric_mean_deltas"]["pass_completion"]["ci95_low"] == 0.25
    assert result["metric_mean_deltas"]["pass_completion"]["ci95_high"] == 0.25


def test_decision_prioritizes_broken_gate():
    status, _ = _decision("no_spatial", {"all_pass": False}, {"delta_loss": {"ci95_low": -1, "ci95_high": 1}})
    assert status == "retain_core"


def test_additive_requires_resolved_improvement():
    report = {"all_pass": True}
    status, _ = _decision("M1", report, {"delta_loss": {"ci95_low": -1, "ci95_high": 0.2}})
    assert status == "research_only_default_off"
