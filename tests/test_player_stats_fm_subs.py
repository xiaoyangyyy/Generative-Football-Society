"""Player stats tracker, FM merge, substitution helpers."""

import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_engine.fm_roster_merge import fm_row_to_abilities, merge_fm_into_roster, normalize_player_name
from src.match_engine.player_match_stats import PlayerMatchStatsTracker
from src.match_engine.micro_events import MicroEvent, MicroEventType


def test_fm_abilities_from_ca():
    ab = fm_row_to_abilities({"ca": 180, "abilities_fm": {"pace": 0.9, "finishing": 0.85}})
    assert ab["pace"] > 0.7
    assert ab["shot"] > 0.6


def test_merge_fm_by_name():
    roster = {
        "team_id": "Brazil",
        "players": [{"name": "Neymar", "player_id": "p1", "abilities": {"tech": 0.5}}],
    }
    fm = [{"name": "Neymar", "ca": 170, "abilities_fm": {"technique": 0.88}}]
    out = merge_fm_into_roster(roster, fm)
    assert out["fm_merge_matched"] == 1
    assert out["players"][0]["abilities"]["tech"] > 0.7


def test_player_tracker_cards():
    trk = PlayerMatchStatsTracker()
    trk.apply_event(
        MicroEvent(60.0, MicroEventType.YELLOW_CARD, "T", player_id="p9")
    )
    assert trk.ensure("p9").yellow_cards == 1
    d = trk.ensure("p9").to_carryover_dict()
    assert d["pass_cmp"] <= 1.0


def test_normalize_name():
    assert normalize_player_name("João Silva") == normalize_player_name("joao silva")
