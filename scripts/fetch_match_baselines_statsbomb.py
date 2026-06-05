#!/usr/bin/env python3
"""Build multi-metric match baselines from StatsBomb open data (WC 2022)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calibration" / "statsbomb_match_baselines.json"


def _q(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def _stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "mean": float(s.mean()) if not s.empty else 0.0,
        "p10": _q(s, 0.10),
        "p25": _q(s, 0.25),
        "p50": _q(s, 0.50),
        "p75": _q(s, 0.75),
        "p90": _q(s, 0.90),
    }


def main() -> int:
    try:
        from statsbombpy import sb
    except ImportError:
        print("Install statsbombpy: pip install statsbombpy", file=sys.stderr)
        return 1

    country, division, season = "International", "FIFA World Cup", "2022"
    print(f"Loading {division} {season} events...")
    ev = sb.competition_events(country=country, division=division, season=season)
    if ev is None or ev.empty:
        print("No events loaded.", file=sys.stderr)
        return 1

    gcols = ["match_id", "team"]
    passes = ev[ev["type"] == "Pass"].copy()
    shots = ev[ev["type"] == "Shot"].copy()
    inter = ev[ev["type"] == "Interception"].copy()
    fouls = ev[ev["type"] == "Foul Committed"].copy()

    pass_att = passes.groupby(gcols).size().rename("passes")
    pass_cmp = passes[passes["pass_outcome"].isna()].groupby(gcols).size().rename("pass_completed")
    through_mask = passes.get("pass_through_ball")
    if through_mask is None:
        through_mask = passes.get("pass_type").astype(str) == "Through Ball"
    else:
        through_mask = through_mask.fillna(False).astype(bool)
    through = passes[through_mask].groupby(gcols).size().rename("through_balls")
    long_pass = passes[pd.to_numeric(passes.get("pass_length"), errors="coerce") >= 30.0].groupby(gcols).size().rename(
        "long_passes"
    )
    cross_col = passes.get("pass_cross")
    if cross_col is not None:
        crosses = passes[cross_col.fillna(False).astype(bool)].groupby(gcols).size().rename("crosses")
    else:
        crosses = passes[passes.get("pass_height").astype(str).str.contains("High", na=False)].groupby(gcols).size().rename(
            "crosses"
        )

    shot_att = shots.groupby(gcols).size().rename("shots")
    sot_outcomes = {"Goal", "Saved", "Saved To Post"}
    sot = shots[shots.get("shot_outcome").astype(str).isin(sot_outcomes)].groupby(gcols).size().rename("shots_on_target")
    goals = shots[shots.get("shot_outcome").astype(str) == "Goal"].groupby(gcols).size().rename("goals")
    xg_col = None
    for cand in ("shot_statsbomb_xg", "xg", "statsbomb_xg"):
        if cand in shots.columns:
            xg_col = cand
            break
    if xg_col is not None:
        team_xg = pd.to_numeric(shots[xg_col], errors="coerce").groupby([shots["match_id"], shots["team"]]).sum()
        team_xg.index.names = gcols
        team_xg = team_xg.rename("xg")
    else:
        team_xg = pd.Series(dtype=float, name="xg")
    body = shots.get("shot_body_part", shots.get("body_part"))
    if body is not None:
        headers = shots[body.astype(str).str.contains("Head", case=False, na=False)].groupby(gcols).size().rename("headers")
    else:
        headers = pd.Series(dtype=float, name="headers")

    interceptions = inter.groupby(gcols).size().rename("interceptions")
    fouls_committed = fouls.groupby(gcols).size().rename("fouls_committed")

    duels = ev[ev["type"] == "Duel"].copy()
    duel_type = duels.get("duel_type", duels.get("type"))
    if duel_type is not None:
        tackles = duels[duel_type.astype(str).str.contains("Tackle", case=False, na=False)].groupby(gcols).size().rename(
            "tackles"
        )
    else:
        tackles = pd.Series(dtype=float, name="tackles")

    cards = ev[ev["type"] == "Bad Behaviour"].copy()
    yc = cards[cards.get("bad_behaviour_card").astype(str).isin(["Yellow Card", "Second Yellow"])].groupby(gcols).size().rename(
        "yellow_cards"
    )
    rc = cards[cards.get("bad_behaviour_card").astype(str).isin(["Red Card", "Second Yellow"])].groupby(gcols).size().rename(
        "red_cards"
    )

    tm = pd.concat(
        [
            pass_att,
            pass_cmp,
            through,
            long_pass,
            crosses,
            shot_att,
            sot,
            goals,
            headers,
            interceptions,
            fouls_committed,
            tackles,
            yc,
            rc,
            team_xg,
        ],
        axis=1,
    ).fillna(0.0)
    for c in tm.columns:
        tm[c] = pd.to_numeric(tm[c], errors="coerce").fillna(0.0)

    match_pass_tot = tm.groupby(level=0)["passes"].transform("sum").clip(lower=1.0)
    tm["possession_share"] = tm["passes"] / match_pass_tot

    tm["pass_completion"] = tm["pass_completed"] / tm["passes"].clip(lower=1.0)
    tm["through_share"] = tm["through_balls"] / tm["passes"].clip(lower=1.0)
    tm["long_pass_share"] = tm["long_passes"] / tm["passes"].clip(lower=1.0)
    tm["interceptions_per_pass"] = tm["interceptions"] / tm["passes"].clip(lower=1.0)
    tm["shots_on_target_rate"] = tm["shots_on_target"] / tm["shots"].clip(lower=1.0)
    if "xg" in tm.columns:
        tm["xg"] = pd.to_numeric(tm["xg"], errors="coerce").fillna(0.0)
        tm["goals_to_xg_ratio"] = tm["goals"] / tm["xg"].clip(lower=0.08)

    metrics = {
        "pass_completion": _stats(tm["pass_completion"]),
        "interceptions_per_pass": _stats(tm["interceptions_per_pass"]),
        "passes_per_team_match": _stats(tm["passes"]),
        "through_share": _stats(tm["through_share"]),
        "long_pass_share": _stats(tm["long_pass_share"]),
        "shots_per_team_match": _stats(tm["shots"]),
        "shots_on_target_rate": _stats(tm["shots_on_target_rate"]),
        "goals_per_team_match": _stats(tm["goals"]),
        "fouls_committed_per_team_match": _stats(tm["fouls_committed"]),
        "yellow_cards_per_team_match": _stats(tm["yellow_cards"]),
        "red_cards_per_team_match": _stats(tm["red_cards"]),
        "possession_share": _stats(tm["possession_share"]),
        "crosses_per_team_match": _stats(tm["crosses"]),
        "headers_per_team_match": _stats(tm["headers"]),
        "tackles_per_team_match": _stats(tm["tackles"]),
    }
    if "xg" in tm.columns:
        metrics["xg_per_team_match"] = _stats(tm["xg"])
        metrics["goals_to_xg_ratio"] = _stats(tm["goals_to_xg_ratio"])

    joint_raw: dict[str, float] = {}
    if len(tm) >= 8:
        joint_raw["possession_passes_correlation"] = float(
            tm["possession_share"].corr(tm["passes"])
        )
        joint_raw["shots_goals_correlation"] = float(tm["shots"].corr(tm["goals"]))

    joint_path = ROOT / "data" / "calibration" / "joint_baselines.json"
    prev_corr: dict[str, Any] = {}
    if joint_path.is_file():
        try:
            prev_corr = json.loads(joint_path.read_text(encoding="utf-8")).get(
                "correlations_reference", {}
            )
        except (json.JSONDecodeError, OSError):
            prev_corr = {}

    def _corr_entry(cid: str, statsbomb_val: float) -> dict[str, Any]:
        prev = prev_corr.get(cid, {})
        if isinstance(prev, dict):
            entry = {
                "statsbomb_mean": statsbomb_val,
                "target_mean": float(prev.get("target_mean", statsbomb_val)),
                "scale": float(prev.get("scale", 0.15 if cid == "possession_passes_correlation" else 0.28)),
            }
            if prev.get("sim_baseline_mean") is not None:
                entry["sim_baseline_mean"] = float(prev["sim_baseline_mean"])
            if prev.get("note"):
                entry["note"] = str(prev["note"])
            return entry
        return {
            "statsbomb_mean": statsbomb_val,
            "target_mean": statsbomb_val,
            "scale": 0.15 if cid == "possession_passes_correlation" else 0.28,
        }

    joint = {
        cid: _corr_entry(cid, val) for cid, val in joint_raw.items()
    }

    payload = {
        "source": f"StatsBomb Open Data — {division} {season} ({country})",
        "n_team_matches": int(len(tm)),
        "metrics": metrics,
        "joint": joint,
        "reasonable_band_default": "p10..p90",
        "sparse_metrics_note": "yellow_cards/red_cards use p10..p90; means near zero — also check |sim-target_mean|",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    soft = {
        "source": payload["source"],
        "n_team_matches": payload["n_team_matches"],
        "soft_constraints": {},
        "correlations_reference": joint,
    }
    if "xg_per_team_match" in metrics:
        soft["soft_constraints"]["micro_xg_per_team_match"] = metrics["xg_per_team_match"]
        soft["soft_constraints"]["goals_to_micro_xg_ratio"] = metrics["goals_to_xg_ratio"]
    joint_path.write_text(json.dumps(soft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Wrote {joint_path}")
    print("shots mean:", round(metrics["shots_per_team_match"]["mean"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
