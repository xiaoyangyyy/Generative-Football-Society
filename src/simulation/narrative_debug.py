"""Structured debug trail for LLM + cognitive + micro stack (tournament runs)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
from src.simulation.runtime import environment_snapshot, env_bool

if TYPE_CHECKING:
    from src.match_engine.state import MicroMatchSummary
    from src.simulation.agent import SocietyAgent


def narrative_debug_enabled() -> bool:
    return env_bool(environment_snapshot(), "MATCH_DEBUG_NARRATIVE", False)


def _debug_log_path() -> Path:
    base = Path(environment_snapshot().get("GFS_BASE_DIR", str(Path.cwd())))
    if not base.is_absolute():
        base = Path.cwd()
    out = base / "outputs" / "narrative_debug.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _write_record(record: Dict[str, Any]) -> None:
    record["ts"] = datetime.now(timezone.utc).isoformat()
    path = _debug_log_path()
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tactical_snapshot(agent: "SocietyAgent") -> Dict[str, Any]:
    vec = dict(getattr(agent, "tactical_vector", {}) or {})
    ctrl = dict(getattr(agent, "tactical_controls", {}) or {})
    cp = getattr(agent, "coach_profile", None)
    keys = (
        "pressing_intensity",
        "risk_budget",
        "line_height",
        "rotation_aggressiveness",
        "through_ball_bias",
        "counter_attack_bias",
        "width_play",
    )
    return {
        "preset": getattr(cp, "preferred_preset", "") if cp else "",
        "formation": getattr(agent, "formation", ""),
        "style_archetype": getattr(agent, "style_archetype", ""),
        "controls": {k: round(float(ctrl.get(k, vec.get(k, 0.5))), 3) for k in keys},
    }


def log_llm_tactics(
    *,
    stage: str,
    home: str,
    away: str,
    agent_home: "SocietyAgent",
    agent_away: "SocietyAgent",
    llm_home: Dict[str, Any],
    llm_away: Dict[str, Any],
) -> None:
    if not narrative_debug_enabled():
        return
    rec = {
        "event": "llm_tactics",
        "stage": stage,
        "home": home,
        "away": away,
        "llm_home": {
            "preset": llm_home.get("tactical_preset", ""),
            "formation": llm_home.get("formation", ""),
            "controls": llm_home.get("controls", {}),
            "hints": list((llm_home.get("tactical_hints") or {}).keys())[:8],
        },
        "llm_away": {
            "preset": llm_away.get("tactical_preset", ""),
            "formation": llm_away.get("formation", ""),
            "controls": llm_away.get("controls", {}),
            "hints": list((llm_away.get("tactical_hints") or {}).keys())[:8],
        },
        "applied_home": _tactical_snapshot(agent_home),
        "applied_away": _tactical_snapshot(agent_away),
    }
    _write_record(rec)


from src.match_engine.calibration.narrative_gate import anomalies_from_match


def _anomalies(
    *,
    s_home: int,
    s_away: int,
    micro: Optional["MicroMatchSummary"],
) -> list[str]:
    if micro is None:
        flags: list[str] = []
        total = s_home + s_away
        if total >= 6:
            flags.append(f"high_total_goals:{total}")
        if s_home >= 5 or s_away >= 5:
            flags.append(f"team_goals_spike:{s_home}-{s_away}")
        return flags
    return anomalies_from_match(
        score_home=s_home,
        score_away=s_away,
        shots_home=int(micro.shots_home),
        shots_away=int(micro.shots_away),
        micro_xg_home=float(micro.micro_xg_home),
        micro_xg_away=float(micro.micro_xg_away),
    )


def log_match_debug(
    *,
    stage: str,
    home: str,
    away: str,
    score_home: int,
    score_away: int,
    xg_home: float,
    xg_away: float,
    agent_home: "SocietyAgent",
    agent_away: "SocietyAgent",
    micro: Optional["MicroMatchSummary"] = None,
    xg_context: str = "",
) -> None:
    if not narrative_debug_enabled():
        return
    anomalies = _anomalies(s_home=score_home, s_away=score_away, micro=micro)
    rec: Dict[str, Any] = {
        "event": "match_result",
        "stage": stage,
        "home": home,
        "away": away,
        "score": [score_home, score_away],
        "macro_xg": [round(xg_home, 3), round(xg_away, 3)],
        "tactics_home": _tactical_snapshot(agent_home),
        "tactics_away": _tactical_snapshot(agent_away),
        "anomalies": anomalies,
        "xg_context": xg_context,
    }
    if micro is not None:
        rec["micro"] = {
            "passes": [micro.passes_home, micro.passes_away],
            "shots": [micro.shots_home, micro.shots_away],
            "sot": [micro.shots_on_target_home, micro.shots_on_target_away],
            "micro_xg": [round(micro.micro_xg_home, 3), round(micro.micro_xg_away, 3)],
            "goals_physics": [micro.goals_physics_home, micro.goals_physics_away],
            "possession_home": round(micro.possession_home, 3),
            "cognitive_plans": len(micro.cognitive_plans),
            "cognitive_tiers": dict(micro.cognitive_tier_usage),
            "phi_integral": [round(micro.phi_integral_home, 3), round(micro.phi_integral_away, 3)],
            "ball_log": micro.ball_log_path,
        }
    _write_record(rec)
    if anomalies:
        print(f"  [DEBUG-NARRATIVE] anomalies={anomalies} | {home} {score_home}-{score_away} {away}")
