from pathlib import Path

import numpy as np

from gfs_studio.bootstrap.application import StudioApplication
from gfs_studio.plugins.training.datasets.manifest import TraceDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_real_cohort_team_time_splits_and_multi_step_rollout(tmp_path):
    app = StudioApplication(PROJECT_ROOT)
    cohort = app.export_cohort(
        [
            {"home": "Brazil", "away": "Argentina", "seed": 42},
            {"home": "France", "away": "England", "seed": 43},
        ],
        tmp_path,
        stop_at_tick=4,
        holdout_teams=["France", "England"],
        validation_time_fraction=0.5,
    )
    assert cohort["rows"] == 8
    assert cohort["matches"][0]["rows"] == 4
    dataset = TraceDataset.from_jsonl(cohort["trace"])
    assert dataset.identity == cohort["trace_sha256"]
    train, test, split = dataset.split_by_teams(["France", "England"])
    assert (split.train_rows, split.validation_rows) == (4, 4)
    train_teams = set(train.group_teams["core:Brazil:Argentina:42"])
    test_teams = set(test.group_teams["core:France:England:43"])
    assert train_teams.isdisjoint(test_teams)
    early, late, time_split = dataset.split_time(validation_fraction=0.5)
    assert (time_split.train_rows, time_split.validation_rows) == (4, 4)
    assert max(early.ticks) < min(late.ticks)
    rollout = app.evaluate_rollout(
        cohort["trace"],
        "latent_wm_rollout_calibrated_candidate.pt",
        horizons=[1, 2, 4],
    )
    assert rollout["eligible_for_deployment"] is False
    assert rollout["cross_match_sequences"] is False
    for horizon, samples in ((1, 8), (2, 6), (4, 2)):
        result = rollout["horizons"][str(horizon)]
        assert result["samples"] == samples
        assert result["finite"] is True
        assert np.isfinite(result["model_mse"])
        assert np.isfinite(result["persistence_mse"])
