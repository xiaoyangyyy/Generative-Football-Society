#!/usr/bin/env python3
"""Train the v7 action-conditioned probabilistic future checkpoint."""

# This executable bootstraps the repository root before importing local packages.
# ruff: noqa: E402

from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.match_engine.world_model.probabilistic import (
    EVENTS,
    fit_event_transition_model,
)


def load_observations():
    statsbomb = pd.read_csv(ROOT / "data/external/statsbomb360/derived/passes.csv")
    sportec = pd.concat(
        [
            pd.read_csv(p)
            for p in sorted(
                (ROOT / "data/external/sportec/derived").glob("*_passes.csv")
            )
        ]
    )
    sportec = sportec[sportec.is_receiver.eq(1)].copy()
    frame = pd.concat([statsbomb, sportec], ignore_index=True)
    bins = np.minimum(
        4, (frame.physics_prior.astype(float).clip(0, 0.9999) * 5).astype(int)
    )
    frame["action"] = "pass_p" + bins.astype(str)
    frame["outcome"] = np.where(
        frame.completion.astype(float) > 0.5, "retain", "turnover"
    )
    shots = [pd.read_csv(ROOT / "data/external/statsbomb360/derived/shots.csv")]
    shots += [
        pd.read_csv(p)
        for p in sorted((ROOT / "data/external/sportec/derived").glob("*_shots.csv"))
    ]
    shot = pd.concat(shots, ignore_index=True)
    shot["action"] = "shot"
    shot["outcome"] = "shot"
    return pd.concat(
        [
            frame[["provider", "match_id", "action", "outcome"]],
            shot[["provider", "match_id", "action", "outcome"]],
        ],
        ignore_index=True,
    )


def main():
    data = load_observations()
    rng = np.random.default_rng(20260724)
    train_parts = []
    test_parts = []
    for _, provider in data.groupby("provider"):
        matches = provider.match_id.astype(str).unique()
        rng.shuffle(matches)
        n = max(1, round(0.2 * len(matches)))
        chosen = set(matches[:n])
        test_parts.append(provider[provider.match_id.astype(str).isin(chosen)])
        train_parts.append(provider[~provider.match_id.astype(str).isin(chosen)])
    train = pd.concat(train_parts)
    test = pd.concat(test_parts)
    model = fit_event_transition_model(
        train.action.to_numpy(), train.outcome.to_numpy()
    )
    probability = np.array(
        [model.predict(a)[o] for a, o in zip(test.action, test.outcome)]
    )
    global_probability = np.array([model.predict("other")[o] for o in test.outcome])
    log_loss = float(-np.log(np.clip(probability, 1e-9, 1)).mean())
    baseline = float(-np.log(np.clip(global_probability, 1e-9, 1)).mean())
    pass_mask = test.action.str.startswith("pass").to_numpy()
    pass_loss = float(-np.log(np.clip(probability[pass_mask], 1e-9, 1)).mean())
    pass_baseline = float(
        -np.log(np.clip(global_probability[pass_mask], 1e-9, 1)).mean()
    )
    report = {
        "version": "7.0.0-candidate",
        "split_policy": "provider-stratified match holdout",
        "train_events": len(train),
        "test_events": len(test),
        "test_matches": int(test.match_id.nunique()),
        "providers": sorted(data.provider.unique()),
        "events": list(EVENTS),
        "log_loss": log_loss,
        "global_baseline_log_loss": baseline,
        "pass_log_loss": pass_loss,
        "pass_baseline_log_loss": pass_baseline,
        "candidate_ready": bool(
            len(test) >= 1000
            and test.match_id.nunique() >= 20
            and log_loss < baseline
            and pass_loss < pass_baseline
        ),
    }
    (ROOT / "data/world_model/probabilistic_v7_candidate.json").write_text(
        json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    target = ROOT / "reports/acceptance/probabilistic_world_v7.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["candidate_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
