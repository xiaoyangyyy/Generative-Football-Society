import hashlib
import json
from pathlib import Path

import pandas as pd

from src.match_engine.receiver_ranker import ReceiverRanker

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_metrica_derived_data_and_ranker_are_sealed_and_valid():
    frames = []
    for game in (1, 2):
        path = ROOT / f"data/external/metrica/pass_candidates_game{game}.csv"
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        assert manifest["derived_sha256"] == _sha256(path)
        frame = pd.read_csv(path)
        assert frame.groupby("event_id").is_receiver.sum().eq(1).all()
        frames.append(frame)
    assert set(frames[0].event_id).isdisjoint(set(frames[1].event_id))
    ranker = ReceiverRanker.load(ROOT / "data/world_model/receiver_ranker.json")
    assert ranker.version == 1
