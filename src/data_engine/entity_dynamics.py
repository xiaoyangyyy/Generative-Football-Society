"""
Continuous entity dynamics for coach / player / team priors.

No hard thresholds or piecewise if/else on roles. Uses:
  - coupled relaxation: dz/dt = -K z + B u  (equilibrium z* = K^{-1} B u)
  - softmax over tactical presets from mental state
  - ability fields: a_k = σ(w_k · φ) with role embedding φ
  - team vector: x_team = tanh(M [x_squad; x_coach])
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.match_engine.math_utils import sigmoid, softmax, tanh_clip
from src.match_engine.tactical_catalog import TACTICAL_KEYS, TACTICAL_PRESETS, infer_archetype_from_text

# Soft reputation prior (σ-bump), not hard attribute overrides
COACH_PRESET_HINTS: Dict[str, str] = {
    "Carlo Ancelotti": "possession_control",
    "Marcelo Bielsa": "gegenpress",
    "Thomas Tuchel": "gegenpress",
    "Didier Deschamps": "catenaccio",
    "Julian Nagelsmann": "gegenpress",
    "Lionel Scaloni": "balanced",
    "Walid Regragui": "low_block_counter",
    "Ralf Rangnick": "gegenpress",
    "Mauricio Pochettino": "high_press",
    "Jesse Marsch": "gegenpress",
    "Vincenzo Montella": "wing_play",
    "Julen Lopetegui": "possession_control",
    "Hervé Renard": "low_block_counter",
    "Giorgos Donis": "balanced",
    "Carlos Queiroz": "low_block_counter",
    "Otto Addo": "balanced",
}

# --- Role geometry (continuous, no if role == ...) ---

ROLE_INDEX: Dict[str, int] = {
    "GK": 0,
    "CB": 1,
    "LB": 2,
    "RB": 3,
    "DM": 4,
    "CM": 5,
    "AM": 6,
    "LM": 7,
    "RM": 8,
    "LW": 9,
    "RW": 10,
    "ST": 11,
}
N_ROLES = len(ROLE_INDEX)

# Player condition dimensions (latent z*, analogous to coach mental)
PLAYER_CONDITION_KEYS: Tuple[str, ...] = (
    "technical_quality",
    "physical_power",
    "composure",
    "pace_threat",
    "vision_playmaking",
    "defensive_intensity",
)

# Playing-channel archetypes (softmax blend → abilities), mirrors tactical presets for coaches
PLAYER_CHANNEL_NAMES: Tuple[str, ...] = (
    "box_finisher",
    "creative_hub",
    "wide_threat",
    "ball_winner",
    "aerial_anchor",
    "progressive_passer",
    "sweeper_keeper",
    "target_forward",
)

# Ability channels for JSON + micro engine (superset of legacy 9-field block)
PLAYER_ABILITY_KEYS: Tuple[str, ...] = (
    "tech",
    "phys",
    "mental",
    "pace",
    "vision",
    "spatial",
    "press",
    "aerial",
    "gk",
    "pass_skill",
    "shot",
    "curve",
    "power",
    "heading",
    "gk_reflex",
    "gk_aerial",
)

PLAYER_CHANNEL_TEMPLATES: Dict[str, Dict[str, float]] = {
    "box_finisher": {
        "tech": 0.82, "phys": 0.72, "mental": 0.68, "pace": 0.78, "vision": 0.62,
        "spatial": 0.60, "press": 0.55, "aerial": 0.58, "gk": 0.0,
        "pass_skill": 0.58, "shot": 0.92, "curve": 0.55, "power": 0.80, "heading": 0.62,
        "gk_reflex": 0.0, "gk_aerial": 0.0,
    },
    "creative_hub": {
        "tech": 0.90, "phys": 0.62, "mental": 0.85, "pace": 0.68, "vision": 0.94,
        "spatial": 0.92, "press": 0.52, "aerial": 0.42, "gk": 0.0,
        "pass_skill": 0.94, "shot": 0.62, "curve": 0.72, "power": 0.55, "heading": 0.40,
        "gk_reflex": 0.0, "gk_aerial": 0.0,
    },
    "wide_threat": {
        "tech": 0.78, "phys": 0.70, "mental": 0.65, "pace": 0.92, "vision": 0.72,
        "spatial": 0.75, "press": 0.62, "aerial": 0.45, "gk": 0.0,
        "pass_skill": 0.72, "shot": 0.70, "curve": 0.68, "power": 0.62, "heading": 0.38,
        "gk_reflex": 0.0, "gk_aerial": 0.0,
    },
    "ball_winner": {
        "tech": 0.68, "phys": 0.82, "mental": 0.72, "pace": 0.70, "vision": 0.58,
        "spatial": 0.62, "press": 0.88, "aerial": 0.65, "gk": 0.0,
        "pass_skill": 0.65, "shot": 0.45, "curve": 0.35, "power": 0.72, "heading": 0.58,
        "gk_reflex": 0.0, "gk_aerial": 0.0,
    },
    "aerial_anchor": {
        "tech": 0.62, "phys": 0.88, "mental": 0.70, "pace": 0.55, "vision": 0.50,
        "spatial": 0.58, "press": 0.72, "aerial": 0.92, "gk": 0.0,
        "pass_skill": 0.55, "shot": 0.48, "curve": 0.30, "power": 0.78, "heading": 0.90,
        "gk_reflex": 0.0, "gk_aerial": 0.0,
    },
    "progressive_passer": {
        "tech": 0.85, "phys": 0.68, "mental": 0.78, "pace": 0.72, "vision": 0.88,
        "spatial": 0.86, "press": 0.65, "aerial": 0.48, "gk": 0.0,
        "pass_skill": 0.90, "shot": 0.55, "curve": 0.58, "power": 0.65, "heading": 0.45,
        "gk_reflex": 0.0, "gk_aerial": 0.0,
    },
    "sweeper_keeper": {
        "tech": 0.72, "phys": 0.75, "mental": 0.80, "pace": 0.58, "vision": 0.70,
        "spatial": 0.78, "press": 0.25, "aerial": 0.55, "gk": 0.94,
        "pass_skill": 0.68, "shot": 0.12, "curve": 0.20, "power": 0.55, "heading": 0.50,
        "gk_reflex": 0.94, "gk_aerial": 0.88,
    },
    "target_forward": {
        "tech": 0.70, "phys": 0.85, "mental": 0.62, "pace": 0.62, "vision": 0.55,
        "spatial": 0.52, "press": 0.58, "aerial": 0.88, "gk": 0.0,
        "pass_skill": 0.58, "shot": 0.78, "curve": 0.40, "power": 0.85, "heading": 0.82,
        "gk_reflex": 0.0, "gk_aerial": 0.0,
    },
}

_CHANNEL_MATRIX = np.zeros((len(PLAYER_CHANNEL_NAMES), len(PLAYER_CONDITION_KEYS)))
for i, cname in enumerate(PLAYER_CHANNEL_NAMES):
    t = PLAYER_CHANNEL_TEMPLATES[cname]
    _CHANNEL_MATRIX[i] = [
        float(t["tech"]),
        float(t["phys"]),
        float(t["mental"]),
        float(t["pace"]),
        float(t["vision"]),
        float(t["press"]),
    ]

# Coach mental dimensions (latent z components)
COACH_MENTAL_KEYS: Tuple[str, ...] = (
    "experience",
    "tactical_knowledge",
    "pressure_handling",
    "adaptability",
    "discipline",
    "motivation",
)

# Preset affinity rows (mental → style), rows sum to interpretable axes
_PRESET_NAMES = tuple(TACTICAL_PRESETS.keys())
_PRESET_MATRIX = np.zeros((len(_PRESET_NAMES), len(COACH_MENTAL_KEYS)))
for i, pname in enumerate(_PRESET_NAMES):
    p = TACTICAL_PRESETS[pname]
    _PRESET_MATRIX[i] = [
        0.35 + 0.45 * p.get("possession_orientation", 0.5),
        0.30 + 0.50 * p.get("pressing_intensity", 0.5),
        0.40 + 0.40 * p.get("low_block", 0.5),
        0.35 + 0.45 * p.get("rotation_aggressiveness", 0.5),
        0.40 + 0.35 * p.get("compactness", 0.5),
        0.35 + 0.45 * p.get("counter_attack", 0.5),
    ]


def role_embedding(role: str, dim: int = 8) -> np.ndarray:
    """Fourier features on role index — smooth in role space."""
    idx = ROLE_INDEX.get(role.upper(), 5)
    t = 2.0 * math.pi * idx / max(N_ROLES, 1)
    feats = [1.0, math.sin(t), math.cos(t), math.sin(2 * t), math.cos(2 * t)]
    # pad / truncate to dim
    v = np.array(feats[:dim], dtype=float)
    if v.size < dim:
        v = np.pad(v, (0, dim - v.size))
    return v / max(1e-9, np.linalg.norm(v))


def _log1p_safe(x: float) -> float:
    return float(math.log1p(max(0.0, x)))


def squad_observables(
    *,
    log_market_value: float,
    international_caps: float,
    height_cm: float,
    squad_max_mv: float,
) -> np.ndarray:
    """Normalized exogenous inputs u ∈ R^4 for player ability field."""
    mv_norm = log_market_value / max(_log1p_safe(squad_max_mv), 1.0)
    caps_norm = tanh_clip(international_caps / 80.0)
    height_norm = tanh_clip((height_cm - 175.0) / 12.0)
    prestige = tanh_clip(0.55 * mv_norm + 0.45 * caps_norm)
    return np.array([mv_norm, caps_norm, height_norm, prestige], dtype=float)


def player_ability_field(
    u: np.ndarray,
    role: str,
    *,
    ability_keys: Optional[Tuple[str, ...]] = None,
) -> Dict[str, float]:
    """
    Ability vector from observation u and role embedding.
    a_i = σ(α_i·u + β_i·φ_role + γ_i).
    Coefficients are fixed smooth weights (calibrated heuristics, not clips).
    """
    keys = ability_keys or (
        "tech",
        "phys",
        "mental",
        "pace",
        "vision",
        "spatial",
        "press",
        "aerial",
        "gk",
    )
    phi = role_embedding(role, dim=8)
    u = np.asarray(u, dtype=float).reshape(-1)
    # coupling matrix W: (n_abilities, 4 + 8)
    W = np.array(
        [
            [1.10, 0.35, 0.15, 0.85, 0.05, 0.02, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.25, 0.20, 0.95, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.45, 0.75, 0.10, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.95, 0.15, 0.05, 0.55, 0.20, -0.15, 0.25, -0.20, 0.30, 0.35, 0.40, 0.15],
            [0.70, 0.50, 0.05, 0.65, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.68, 0.48, 0.05, 0.62, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.55, 0.25, 0.00, 0.50, 0.15, 0.10, 0.20, 0.10, 0.05, 0.05, 0.05, 0.00],
            [0.20, 0.15, 0.55, 0.35, 0.45, 0.40, 0.35, 0.10, 0.05, 0.05, 0.05, 0.25],
            [-2.5, 0.10, 0.20, 0.30, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        dtype=float,
    )
    W = W[: len(keys)]
    x = np.concatenate([u, phi])
    logits = W @ x
    gk_idx = ROLE_INDEX.get(role.upper(), 5)
    logits[-1] = logits[-1] + 1.8 * math.exp(-0.5 * ((gk_idx - 0) / 1.2) ** 2)

    out: Dict[str, float] = {}
    for k, lg in zip(keys, logits):
        out[k] = float(sigmoid(lg))
    return out


def player_exogenous_inputs(
    *,
    log_market_value: float,
    international_caps: float,
    height_cm: float,
    squad_max_mv: float,
    age: Optional[float] = None,
) -> np.ndarray:
    """
    u_player ∈ R^5 — mirrors coach exogenous block.
    [mv_norm, caps_norm, height_norm, prestige, age_prime]
    """
    u4 = squad_observables(
        log_market_value=log_market_value,
        international_caps=international_caps,
        height_cm=height_cm,
        squad_max_mv=squad_max_mv,
    )
    age_val = 26.0 if age is None else float(age)
    age_prime = tanh_clip((27.0 - age_val) / 8.0)
    return np.array([u4[0], u4[1], u4[2], u4[3], age_prime], dtype=float)


def player_condition_equilibrium(u: np.ndarray, role: str) -> Dict[str, float]:
    """z* = K_p^{-1} B_p u with role-coupled injection on defensive / gk rows."""
    u = np.asarray(u, dtype=float).reshape(-1)
    K = np.array(
        [
            [2.0, -0.08, 0.0, 0.0, 0.0, 0.0],
            [-0.06, 1.9, -0.10, 0.0, 0.0, 0.0],
            [-0.12, 0.06, 2.2, 0.0, 0.0, 0.0],
            [-0.05, 0.0, 0.0, 1.85, 0.0, 0.0],
            [0.0, 0.0, -0.08, 0.10, 2.05, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.12, 2.15],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1.25, 0.30, 0.10, 0.55, 0.12],
            [0.20, 0.85, 0.05, 0.35, 0.08],
            [0.35, 0.25, 1.10, 0.40, 0.15],
            [0.90, 0.10, 0.05, 0.45, 0.22],
            [0.40, 0.35, 0.45, 0.30, 0.10],
            [0.15, 0.20, 0.25, 0.20, 0.05],
        ],
        dtype=float,
    )
    z_star = np.linalg.solve(K, B @ u)
    phi = role_embedding(role, dim=8)
    gk_bump = float(phi[0]) * 1.2 if ROLE_INDEX.get(role.upper(), 5) == 0 else 0.0
    z_star[5] = z_star[5] + gk_bump
    z_star[3] = z_star[3] + 0.35 * float(phi[1])
    return {k: float(sigmoid(z)) for k, z in zip(PLAYER_CONDITION_KEYS, z_star)}


def player_channel_affinities(
    condition: Dict[str, float],
    role: str,
    *,
    temperature: float = 0.9,
) -> Dict[str, float]:
    z = np.array([condition.get(k, 0.5) for k in PLAYER_CONDITION_KEYS], dtype=float)
    logits = _CHANNEL_MATRIX @ z
    phi = role_embedding(role, dim=8)
    for i, cname in enumerate(PLAYER_CHANNEL_NAMES):
        if cname == "sweeper_keeper":
            logits[i] += 1.1 * float(phi[0])
        if cname == "wide_threat" and ROLE_INDEX.get(role.upper(), 5) in (9, 10, 7, 8):
            logits[i] += 0.55
        if cname == "aerial_anchor" and ROLE_INDEX.get(role.upper(), 5) in (1,):
            logits[i] += 0.45
    p = softmax(logits / temperature)
    return {name: float(v) for name, v in zip(PLAYER_CHANNEL_NAMES, p)}


def blend_abilities_from_channels(affinities: Dict[str, float]) -> Dict[str, float]:
    out = {k: 0.0 for k in PLAYER_ABILITY_KEYS}
    for cname, w in affinities.items():
        tmpl = PLAYER_CHANNEL_TEMPLATES.get(cname)
        if not tmpl:
            continue
        for key in PLAYER_ABILITY_KEYS:
            out[key] = out[key] + w * float(tmpl.get(key, tmpl.get("tech", 0.5)))
    return {k: float(sigmoid(2.0 * v - 1.0)) for k, v in out.items()}


def dominant_channel(affinities: Dict[str, float]) -> str:
    return max(affinities.items(), key=lambda kv: kv[1])[0]


def build_player_dynamics_payload(
    *,
    role: str,
    log_market_value: float,
    international_caps: float,
    height_cm: float,
    squad_max_mv: float,
    age: Optional[float] = None,
) -> Dict[str, Any]:
    u = player_exogenous_inputs(
        log_market_value=log_market_value,
        international_caps=international_caps,
        height_cm=height_cm,
        squad_max_mv=squad_max_mv,
        age=age,
    )
    condition = player_condition_equilibrium(u, role)
    aff = player_channel_affinities(condition, role)
    abilities = blend_abilities_from_channels(aff)
    return {
        "condition": condition,
        "channel_affinities": aff,
        "primary_channel": dominant_channel(aff),
        "dynamics_input": {f"u{i}": float(v) for i, v in enumerate(u)},
        "role_embedding": role_embedding(role, dim=8).tolist(),
        "abilities": abilities,
        "dynamics_model": "entity_dynamics_v1",
    }


def coach_exogenous_inputs(
    *,
    tenure_years: float,
    fifa_ranking: float,
    reputation_prior: float = 0.0,
    style_desc: str = "",
) -> np.ndarray:
    """
    u_coach = [τ, π_pressure, ρ_rep, σ_style]
    π_pressure = tanh((25-rank)/8), τ = tanh(tenure/6), etc.
    """
    tau = tanh_clip(tenure_years / 6.0)
    pressure = tanh_clip((25.0 - fifa_ranking) / 8.0)
    rep = tanh_clip(reputation_prior)
    style_signal = tanh_clip(0.15 * len(style_desc) / 200.0 + 0.25 * rep)
    return np.array([tau, pressure, rep, style_signal], dtype=float)


def coach_mental_equilibrium(u: np.ndarray) -> Dict[str, float]:
    """
    Coupled linear relaxation equilibrium z* = K^{-1} B u.
    z components map to COACH_MENTAL_KEYS via sigmoid readout.
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    K = np.array(
        [
            [2.2, -0.15, 0.0, 0.0, 0.0, 0.0],
            [-0.10, 2.0, -0.12, 0.0, 0.0, 0.0],
            [-0.35, 0.05, 2.4, 0.0, 0.0, 0.0],
            [0.0, 0.10, 0.0, 1.8, 0.0, 0.0],
            [0.0, 0.08, 0.0, 0.05, 1.9, 0.0],
            [-0.12, 0.0, -0.20, 0.0, 0.0, 2.1],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [1.4, 0.0, 0.35, 0.05],
            [0.55, 0.0, 0.50, 0.10],
            [0.20, -0.85, 0.15, 0.0],
            [0.40, 0.0, 0.10, 0.15],
            [0.25, 0.0, 0.20, 0.05],
            [0.35, -0.25, 0.15, 0.20],
        ],
        dtype=float,
    )
    z_star = np.linalg.solve(K, B @ u)
    return {k: float(sigmoid(z)) for k, z in zip(COACH_MENTAL_KEYS, z_star)}


def coach_reputation_prior(coach_name: str, known_hints: Mapping[str, str]) -> float:
    """Smooth prior from hint table: σ(2·𝟙_known), no hard add to experience."""
    if coach_name in known_hints:
        return float(sigmoid(1.6))
    return float(sigmoid(-0.4))


def preset_affinities(
    mental: Dict[str, float],
    style_desc: str = "",
    *,
    temperature: float = 0.85,
) -> Dict[str, float]:
    """Softmax affinity over all tactical presets."""
    z = np.array([mental.get(k, 0.5) for k in COACH_MENTAL_KEYS], dtype=float)
    logits = _PRESET_MATRIX @ z
    arch = infer_archetype_from_text(style_desc)
    if arch in TACTICAL_PRESETS:
        j = _PRESET_NAMES.index(arch)
        logits[j] = logits[j] + 0.65
    p = softmax(logits / temperature)
    return {name: float(v) for name, v in zip(_PRESET_NAMES, p)}


def blend_preset_from_affinities(affinities: Dict[str, float]) -> Dict[str, float]:
    """Continuous tactical vector = Σ_k π_k · preset_k."""
    out = {k: 0.0 for k in TACTICAL_KEYS}
    for pname, w in affinities.items():
        if pname not in TACTICAL_PRESETS:
            continue
        for key in TACTICAL_KEYS:
            out[key] = out[key] + w * float(TACTICAL_PRESETS[pname].get(key, 0.5))
    return {k: float(sigmoid(2.0 * v - 1.0)) for k, v in out.items()}


def dominant_preset(affinities: Dict[str, float]) -> str:
    return max(affinities.items(), key=lambda kv: kv[1])[0]


def coach_authority_dynamics(mental: Dict[str, float]) -> float:
    """σ(w·z) — bounded without clip()."""
    z = np.array([mental.get(k, 0.5) for k in COACH_MENTAL_KEYS], dtype=float)
    w = np.array([0.35, 0.28, 0.22, 0.08, 0.18, 0.12], dtype=float)
    b = -0.08
    return float(sigmoid(w @ z + b))


def team_state_vector(
    squad_abilities: List[Dict[str, float]],
    coach_mental: Dict[str, float],
    *,
    fifa_ranking: float = 50.0,
    squad_conditions: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Aggregate team latent x_T ∈ R^5 from squad + coach (steady summary, not discrete tiers).
    Components: attack, defense, press, morale_field, institutional_pressure
    """
    if not squad_abilities:
        z = np.zeros(5)
    else:
        tech = np.mean([v.get("tech", 0.5) for v in squad_abilities])
        pace = np.mean([v.get("pace", 0.5) for v in squad_abilities])
        press = np.mean([v.get("press", 0.5) for v in squad_abilities])
        aerial = np.mean([v.get("aerial", 0.5) for v in squad_abilities])
        gk = np.mean([v.get("gk", 0.0) for v in squad_abilities])
        attack = tanh_clip(0.55 * tech + 0.45 * pace)
        defense = tanh_clip(0.50 * aerial + 0.50 * gk + 0.15)
        press_f = tanh_clip(press)
        if squad_conditions:
            comp = np.mean([c.get("composure", 0.5) for c in squad_conditions])
            pq = np.mean([c.get("technical_quality", 0.5) for c in squad_conditions])
            morale_player = tanh_clip(0.55 * comp + 0.45 * pq)
        else:
            morale_player = tanh_clip(tech)
        morale_f = tanh_clip(
            0.30 * coach_mental.get("motivation", 0.5)
            + 0.25 * coach_mental.get("discipline", 0.5)
            + 0.20 * coach_mental.get("experience", 0.5)
            + 0.25 * morale_player
        )
        inst_p = tanh_clip((25.0 - fifa_ranking) / 10.0)
        z = np.array([attack, defense, press_f, morale_f, inst_p], dtype=float)

    keys = ("attack", "defense", "press", "morale_field", "institutional_pressure")
    return {k: float(v) for k, v in zip(keys, z)}


def latent_logit_from_unit_interval(x: float, eps: float = 1e-4) -> float:
    x = min(1.0 - eps, max(eps, float(x)))
    return float(math.log(x / (1.0 - x)))
