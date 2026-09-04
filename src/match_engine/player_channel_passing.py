"""Channel affinities → passing utility biases (not only blended abilities)."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.match_engine.state import PlayerAffectiveState

# Per-channel additive utility on pass kind
_CHANNEL_PASS_BIAS: Dict[str, Dict[str, float]] = {
    "box_finisher": {"short": 0.05, "through": 0.22, "long": -0.08},
    "creative_hub": {"short": 0.18, "through": 0.42, "long": 0.05, "wall": 0.12},
    "wide_threat": {"short": 0.08, "through": 0.15, "long": 0.38},
    "ball_winner": {"short": 0.28, "through": -0.05, "long": -0.1, "wall": 0.08},
    "aerial_anchor": {"short": 0.12, "long": 0.25, "through": -0.12},
    "progressive_passer": {"short": 0.22, "through": 0.35, "long": 0.12},
    "sweeper_keeper": {"short": 0.35, "long": 0.2, "through": -0.15},
    "target_forward": {"short": 0.1, "long": 0.32, "through": 0.18},
}


def channel_pass_utility_boost(
    carrier: "PlayerAffectiveState",
    kind: str,
) -> float:
    aff: Dict[str, float] = getattr(carrier, "channel_affinities", None) or {}
    if not aff:
        primary = getattr(carrier, "primary_channel", "") or ""
        if primary and primary in _CHANNEL_PASS_BIAS:
            return float(_CHANNEL_PASS_BIAS[primary].get(kind, 0.0))
        return 0.0
    boost = 0.0
    for ch, w in aff.items():
        biases = _CHANNEL_PASS_BIAS.get(ch, {})
        boost += float(w) * float(biases.get(kind, 0.0))
    return boost


def blend_pass_risk_from_channels(
    carrier: "PlayerAffectiveState",
    team_tac: Dict[str, float],
) -> Dict[str, float]:
    """Adjust team tactical pass-related knobs from squad channel centroid."""
    aff: Dict[str, float] = getattr(carrier, "channel_affinities", None) or {}
    if not aff:
        return team_tac
    creative = aff.get("creative_hub", 0) + aff.get("progressive_passer", 0)
    vertical = aff.get("box_finisher", 0) + aff.get("wide_threat", 0)
    out = dict(team_tac)
    out["through_ball_bias"] = float(
        0.55 * out.get("through_ball_bias", 0.5) + 0.45 * creative
    )
    out["long_ball_bias"] = float(0.55 * out.get("long_ball_bias", 0.5) + 0.45 * vertical)
    out["build_up_short"] = float(0.6 * out.get("build_up_short", 0.5) + 0.4 * aff.get("ball_winner", 0))
    return out
