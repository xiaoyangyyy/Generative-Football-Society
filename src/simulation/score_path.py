"""Phase 4 — single official score path; macro λ is narrative-only when physics is on."""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, Tuple
from src.simulation.runtime import environment_snapshot, env_bool


class ScorePathMode(str, Enum):
    """How official match goals are produced."""

    PHYSICS_OFFICIAL = "physics_official"  # micro shot/aerial physics → score
    MICRO_REPLAY = "micro_replay"  # macro samples score; micro replays anchored
    MACRO_UNIFIED = "macro_unified"  # macro λ (+ optional micro xG blend) → Poisson
    POISSON_LEGACY = "poisson_legacy"  # status-vector Poisson only


def _env_truthy(name: str) -> bool:
    return env_bool(environment_snapshot(), name, False)


def _micro_enabled() -> bool:
    return _env_truthy("MATCH_MICRO")


def resolve_score_path_mode() -> ScorePathMode:
    if _env_truthy("MATCH_LEGACY_POISSON"):
        return ScorePathMode.POISSON_LEGACY
    if not _micro_enabled():
        return ScorePathMode.MACRO_UNIFIED
    raw = environment_snapshot().get("MATCH_MICRO_SCORE", "").strip().lower()
    if raw in ("0", "false", "no"):
        return ScorePathMode.MICRO_REPLAY
    return ScorePathMode.PHYSICS_OFFICIAL


def physics_official_enabled() -> bool:
    return resolve_score_path_mode() == ScorePathMode.PHYSICS_OFFICIAL


def macro_narrative_only() -> bool:
    """Macro λ/xG may shape calendars and agent narrative but not official goals."""
    return resolve_score_path_mode() in (
        ScorePathMode.PHYSICS_OFFICIAL,
        ScorePathMode.MICRO_REPLAY,
    )


def xg_supplement_allowed() -> bool:
    """Poisson top-up of physics goals — forbidden on physics-official path."""
    if physics_official_enabled():
        return False
    return _env_truthy("XG_SUPPLEMENT")


def scheduled_shots_allowed() -> bool:
    raw = environment_snapshot().get("MATCH_SCHEDULED_SHOTS", "").strip().lower()
    if raw in ("0", "false", "off", "none", "pure"):
        return False
    if raw in ("1", "true", "yes", "calibrated"):
        return True
    if physics_official_enabled():
        return False
    return raw != ""


def map_micro_score_to_fixture(
    home_micro: str,
    fixture_home: str,
    goals_micro_home: int,
    goals_micro_away: int,
) -> Tuple[int, int]:
    if home_micro == fixture_home:
        return int(goals_micro_home), int(goals_micro_away)
    return int(goals_micro_away), int(goals_micro_home)


def map_micro_xg_to_fixture(
    home_micro: str,
    fixture_home: str,
    xg_micro_home: float,
    xg_micro_away: float,
) -> Tuple[float, float]:
    if home_micro == fixture_home:
        return float(xg_micro_home), float(xg_micro_away)
    return float(xg_micro_away), float(xg_micro_home)


def finalize_official_score_from_micro(
    micro_summary: Any,
    *,
    home_micro: str,
    fixture_home: str,
    macro_xg_prior_home: float,
    macro_xg_prior_away: float,
) -> Tuple[int, int, float, float, Dict[str, Any]]:
    """
    Official score = micro physics goals only.
    Reported xG = micro integral; macro prior kept for narrative/logging.
    """
    gh = int(getattr(micro_summary, "goals_micro_home", 0))
    ga = int(getattr(micro_summary, "goals_micro_away", 0))
    xgh = float(getattr(micro_summary, "micro_xg_home", 0.0))
    xga = float(getattr(micro_summary, "micro_xg_away", 0.0))
    s1, s2 = map_micro_score_to_fixture(home_micro, fixture_home, gh, ga)
    xg1, xg2 = map_micro_xg_to_fixture(home_micro, fixture_home, xgh, xga)
    meta: Dict[str, Any] = {
        "source": "physics_official",
        "score_path": ScorePathMode.PHYSICS_OFFICIAL.value,
        "macro_xg_prior": [float(macro_xg_prior_home), float(macro_xg_prior_away)],
        "micro_xg": [xgh, xga],
        "goals_physics": [
            int(getattr(micro_summary, "goals_physics_home", gh)),
            int(getattr(micro_summary, "goals_physics_away", ga)),
        ],
        "xg_supplement_applied": bool(getattr(micro_summary, "xg_supplement_applied", False)),
    }
    return s1, s2, xg1, xg2, meta


def score_path_label(mode: ScorePathMode | None = None) -> str:
    mode = mode or resolve_score_path_mode()
    return {
        ScorePathMode.PHYSICS_OFFICIAL: "micro_physics_first",
        ScorePathMode.MICRO_REPLAY: "micro_replay",
        ScorePathMode.MACRO_UNIFIED: "macro_unified",
        ScorePathMode.POISSON_LEGACY: "poisson_legacy",
    }[mode]
