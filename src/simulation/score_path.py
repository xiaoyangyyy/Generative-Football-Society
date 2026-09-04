"""Phase 4 — single official score path; macro λ is narrative-only when physics is on."""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from numbers import Integral, Real
from typing import Any, Dict, Tuple
from src.simulation.runtime import environment_snapshot, env_bool


class ScorePathMode(str, Enum):
    """How official match goals are produced."""

    PHYSICS_OFFICIAL = "physics_official"  # micro shot/aerial physics → score
    MICRO_REPLAY = "micro_replay"  # macro samples score; micro replays anchored
    MACRO_UNIFIED = "macro_unified"  # macro λ (+ optional micro xG blend) → Poisson
    POISSON_LEGACY = "poisson_legacy"  # status-vector Poisson only


class OfficialScoreIntegrityError(ValueError):
    """The physics-official result cannot be proven from its micro summary."""


def _required_summary_value(summary: Any, field: str) -> Any:
    if isinstance(summary, Mapping):
        if field not in summary:
            raise OfficialScoreIntegrityError(
                f"Physics-official summary is missing {field}"
            )
        return summary[field]
    if not hasattr(summary, field):
        raise OfficialScoreIntegrityError(
            f"Physics-official summary is missing {field}"
        )
    return getattr(summary, field)


def _official_goal(summary: Any, field: str) -> int:
    value = _required_summary_value(summary, field)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise OfficialScoreIntegrityError(
            f"Physics-official summary has invalid {field}"
        )
    return int(value)


def _official_xg(summary: Any, field: str) -> float:
    value = _required_summary_value(summary, field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OfficialScoreIntegrityError(
            f"Physics-official summary has invalid {field}"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise OfficialScoreIntegrityError(
            f"Physics-official summary has invalid {field}"
        )
    return result


def validate_official_xg_prior(value: Any, field: str) -> float:
    """Return one finite non-negative macro prior used by physics scoring."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OfficialScoreIntegrityError(
            f"Physics-official score has invalid {field}"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise OfficialScoreIntegrityError(
            f"Physics-official score has invalid {field}"
        )
    return result


def validate_physics_official_summary(summary: Any) -> Dict[str, Any]:
    """Return complete score evidence or reject the micro result."""
    gh = _official_goal(summary, "goals_micro_home")
    ga = _official_goal(summary, "goals_micro_away")
    physics_gh = _official_goal(summary, "goals_physics_home")
    physics_ga = _official_goal(summary, "goals_physics_away")
    xgh = _official_xg(summary, "micro_xg_home")
    xga = _official_xg(summary, "micro_xg_away")
    supplement = _required_summary_value(summary, "xg_supplement_applied")
    if not isinstance(supplement, bool):
        raise OfficialScoreIntegrityError(
            "Physics-official summary has invalid xg_supplement_applied"
        )
    if supplement:
        raise OfficialScoreIntegrityError(
            "Physics-official score cannot contain an xG supplement"
        )
    if (gh, ga) != (physics_gh, physics_ga):
        raise OfficialScoreIntegrityError(
            "Physics-official goals do not match physics goal evidence"
        )
    return {
        "source": ScorePathMode.PHYSICS_OFFICIAL.value,
        "goals_micro": [gh, ga],
        "goals_physics": [physics_gh, physics_ga],
        "micro_xg": [xgh, xga],
        "xg_supplement_applied": supplement,
    }


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
    evidence = validate_physics_official_summary(micro_summary)
    gh, ga = evidence["goals_micro"]
    physics_gh, physics_ga = evidence["goals_physics"]
    xgh, xga = evidence["micro_xg"]
    prior_home = validate_official_xg_prior(
        macro_xg_prior_home, "macro_xg_prior_home"
    )
    prior_away = validate_official_xg_prior(
        macro_xg_prior_away, "macro_xg_prior_away"
    )
    s1, s2 = map_micro_score_to_fixture(home_micro, fixture_home, gh, ga)
    xg1, xg2 = map_micro_xg_to_fixture(home_micro, fixture_home, xgh, xga)
    meta: Dict[str, Any] = {
        "source": "physics_official",
        "score_path": ScorePathMode.PHYSICS_OFFICIAL.value,
        "macro_xg_prior": [prior_home, prior_away],
        "micro_xg": [xgh, xga],
        "goals_physics": [physics_gh, physics_ga],
        "xg_supplement_applied": evidence["xg_supplement_applied"],
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
