"""Versioned snapshots for simulation-owned dynamic tournament state."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data_engine.coach_loader import CoachProfile
from src.simulation.narrative_events import NarrativeEvent
from src.simulation.psychological_state import PsychologicalState


WORLD_STATE_VERSION = 1

# Static identity, priors and model services are rebuilt from inputs already bound by
# the run manifest. These fields are the simulation-owned values that can evolve.
AGENT_DYNAMIC_FIELDS = (
    "status_score", "tier", "personality", "roles", "W_h", "momentum",
    "rivalry_database", "media_filter", "W_x", "reflection_diary",
    "psychology_profile", "formation", "style_desc", "style_archetype",
    "semantic_memory", "episodic_memory", "procedural_memory",
    "decision_memory", "memory_event_log", "memory_clock", "beliefs",
    "llm_reflection_audit", "fatigue", "injury_list", "injury_load",
    "readiness", "tactical_controls", "coach_authority", "icon_influence",
    "team_cohesion", "conflict_heat", "referee_trust",
    "referee_grievance", "social_narrative_state", "appraisal_state",
    "emotion_profile", "coping_profile", "z_state", "team_dynamics",
    "_tactical_vector", "tactical_vector", "_tactical_preset_locked",
    "_prematch_world_model_audit", "psychological_state",
)
AGENT_OPTIONAL_FIELDS = frozenset({
    "team_dynamics", "_tactical_vector", "tactical_vector",
    "_tactical_preset_locked", "_prematch_world_model_audit",
    "psychological_state",
})
AGENT_REQUIRED_FIELDS = frozenset(AGENT_DYNAMIC_FIELDS) - AGENT_OPTIONAL_FIELDS
AGENT_REBUILT_FIELDS = frozenset({
    "name", "team_name", "stats", "region", "media_exposure",
    "coach_profile", "coach_name", "random_root_seed", "memory_max_episodic",
    "memory_decay_lambda", "tau_e", "tau_c", "lambda_m", "beta_m",
    "gamma_retrieval", "eta_retrieval", "rho_b", "k_alignment",
    "alpha_conf", "s_b", "c0", "delta_max", "rho_s",
    "state_noise_sigma", "eps", "squad_carryover",
    "_roster_carryover_snapshot",
})

_TYPE_KEY = "__gfs_snapshot_type__"


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("World snapshot contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, np.ndarray):
        return {
            _TYPE_KEY: "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _encode(value.tolist()),
        }
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "values": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("World snapshot mappings require string keys")
        encoded = {key: _encode(item) for key, item in value.items()}
        if _TYPE_KEY in encoded:
            return {_TYPE_KEY: "mapping", "values": encoded}
        return encoded
    if isinstance(value, PsychologicalState):
        return {_TYPE_KEY: "psychological_state", "values": _encode(asdict(value))}
    if isinstance(value, NarrativeEvent):
        return {_TYPE_KEY: "narrative_event", "values": _encode(asdict(value))}
    if is_dataclass(value):
        raise TypeError(f"Unsupported snapshot dataclass: {type(value).__name__}")
    raise TypeError(f"Unsupported world snapshot value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("World snapshot contains a non-finite float")
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported encoded world snapshot value: {type(value).__name__}")
    kind = value.get(_TYPE_KEY)
    if kind is None:
        return {key: _decode(item) for key, item in value.items()}
    if set(value) != {_TYPE_KEY, "values"} and kind != "ndarray":
        raise ValueError(f"Malformed world snapshot type: {kind!r}")
    if kind == "tuple":
        return tuple(_decode(item) for item in value["values"])
    if kind == "mapping":
        decoded = value["values"]
        if not isinstance(decoded, dict):
            raise ValueError("Malformed mapping snapshot")
        return {key: _decode(item) for key, item in decoded.items()}
    if kind == "psychological_state":
        return PsychologicalState(**_decode(value["values"]))
    if kind == "narrative_event":
        return NarrativeEvent(**_decode(value["values"]))
    if kind == "ndarray":
        if set(value) != {_TYPE_KEY, "dtype", "shape", "values"}:
            raise ValueError("Malformed ndarray snapshot")
        try:
            dtype = np.dtype(str(value["dtype"]))
        except TypeError as exc:
            raise ValueError("Invalid ndarray snapshot dtype") from exc
        if dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("Unsupported ndarray snapshot dtype")
        array = np.asarray(_decode(value["values"]), dtype=dtype)
        if dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError("World snapshot ndarray contains non-finite values")
        shape = tuple(int(item) for item in value["shape"])
        if array.shape != shape:
            raise ValueError("World snapshot ndarray shape mismatch")
        return array
    raise ValueError(f"Unknown world snapshot type: {kind!r}")


def _snapshot_agent(agent: Any) -> dict[str, Any]:
    unknown = set(vars(agent)) - set(AGENT_DYNAMIC_FIELDS) - set(
        AGENT_REBUILT_FIELDS
    )
    if unknown:
        raise ValueError(
            "Unclassified mutable Agent fields: " + ", ".join(sorted(unknown))
        )
    state = {
        field: _encode(getattr(agent, field))
        for field in AGENT_DYNAMIC_FIELDS
        if hasattr(agent, field)
    }
    coach = getattr(agent, "coach_profile", None)
    coach_state = None
    if coach is not None:
        if not isinstance(coach, CoachProfile):
            raise TypeError("Unsupported coach profile in world snapshot")
        coach_state = {
            "team_id": coach.team_id,
            "preferred_preset": coach.preferred_preset,
            "preset_affinities": _encode(coach.preset_affinities),
        }
    return {
        "team_name": str(agent.team_name),
        "random_root_seed": int(agent.random_root_seed),
        "state": state,
        "coach_state": coach_state,
    }


def snapshot_world_state(manager: Any) -> dict[str, Any]:
    """Capture causal in-memory state; external carryover stays separately bound."""
    world = manager.world
    return {
        "schema_version": WORLD_STATE_VERSION,
        "current_date": pd.Timestamp(world.current_date).isoformat(),
        "feed_posts": _encode(world.feed.posts),
        "agents": {
            str(name): _snapshot_agent(agent)
            for name, agent in sorted(world.agents.items())
        },
        "dialogue": {
            "market_step": int(manager.dialogue_engine.market_step),
            "topic_market": _encode(manager.dialogue_engine.topic_market),
        },
        "narrative_history": _encode(manager.narrative_event_bus.history),
    }


def validate_world_state(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != WORLD_STATE_VERSION:
        raise ValueError("Invalid tournament world-state snapshot version")
    required = {
        "schema_version", "current_date", "feed_posts", "agents",
        "dialogue", "narrative_history",
    }
    if set(payload) != required or not isinstance(payload["agents"], dict):
        raise ValueError("Invalid tournament world-state snapshot structure")
    if not isinstance(payload["current_date"], str):
        raise ValueError("Invalid tournament world-state date")
    try:
        parsed_date = pd.to_datetime(payload["current_date"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid tournament world-state date") from exc
    if pd.isna(parsed_date):
        raise ValueError("Invalid tournament world-state date")
    dialogue = payload["dialogue"]
    if not isinstance(dialogue, dict) or set(dialogue) != {"market_step", "topic_market"}:
        raise ValueError("Invalid tournament dialogue snapshot")
    if (
        isinstance(dialogue["market_step"], bool)
        or not isinstance(dialogue["market_step"], int)
        or dialogue["market_step"] < 0
    ):
        raise ValueError("Invalid tournament dialogue market step")
    if not isinstance(_decode(payload["feed_posts"]), list):
        raise ValueError("Invalid tournament social-feed snapshot")
    if not isinstance(_decode(dialogue["topic_market"]), dict):
        raise ValueError("Invalid tournament topic-market snapshot")
    if not isinstance(_decode(payload["narrative_history"]), list):
        raise ValueError("Invalid tournament narrative-history snapshot")
    for name, record in payload["agents"].items():
        if not isinstance(name, str) or not name or not isinstance(record, dict):
            raise ValueError("Invalid tournament agent snapshot")
        if set(record) != {
            "team_name", "random_root_seed", "state", "coach_state",
        } or record["team_name"] != name:
            raise ValueError("Invalid tournament agent snapshot")
        seed = record["random_root_seed"]
        state = record["state"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Invalid tournament agent random-world snapshot")
        if not isinstance(state, dict) or any(
            field not in AGENT_DYNAMIC_FIELDS for field in state
        ) or not AGENT_REQUIRED_FIELDS.issubset(state):
            raise ValueError("Invalid tournament agent dynamic state")
        for value in state.values():
            _decode(value)
        coach_state = record["coach_state"]
        if coach_state is not None:
            if not isinstance(coach_state, dict) or set(coach_state) != {
                "team_id", "preferred_preset", "preset_affinities",
            }:
                raise ValueError("Invalid tournament coach snapshot")
            if (
                not isinstance(coach_state["team_id"], str)
                or not isinstance(coach_state["preferred_preset"], str)
            ):
                raise ValueError("Invalid tournament coach identity")
            if not isinstance(_decode(coach_state["preset_affinities"]), dict):
                raise ValueError("Invalid tournament coach affinities")
    return payload


def preflight_world_state(manager: Any, payload: Any) -> dict[str, Any]:
    """Validate the complete snapshot against the live world without mutation."""
    snapshot = validate_world_state(payload)
    world = manager.world
    expected_agents = set(world.agents)
    stored_agents = set(snapshot["agents"])
    if stored_agents != expected_agents:
        raise ValueError("Tournament world-state agent identity mismatch")

    # Validate every live identity before mutating any in-memory world object.
    for name, record in snapshot["agents"].items():
        agent = world.agents[name]
        if record["team_name"] != agent.team_name:
            raise ValueError("Tournament agent name mismatch")
        if int(record["random_root_seed"]) != int(agent.random_root_seed):
            raise ValueError("Tournament agent random-world mismatch")
        state = record["state"]
        coach_state = record["coach_state"]
        coach = getattr(agent, "coach_profile", None)
        if coach_state is None and coach is not None:
            raise ValueError("Tournament coach snapshot is missing")
        if coach_state is not None:
            if not isinstance(coach, CoachProfile):
                raise ValueError("Tournament coach profile is unavailable")
            if coach_state.get("team_id") != coach.team_id:
                raise ValueError("Tournament coach identity mismatch")
        if any(
            hasattr(agent, field) and field not in state
            for field in AGENT_OPTIONAL_FIELDS
        ):
            raise ValueError("Tournament agent optional-state presence mismatch")
    return snapshot


def restore_world_state(manager: Any, payload: Any) -> None:
    snapshot = preflight_world_state(manager, payload)
    world = manager.world
    world.current_date = pd.to_datetime(snapshot["current_date"])
    world.feed.posts = _decode(snapshot["feed_posts"])
    for name, record in snapshot["agents"].items():
        agent = world.agents[name]
        for field, value in record["state"].items():
            setattr(agent, field, _decode(value))
        coach_state = record["coach_state"]
        if coach_state is not None:
            coach = agent.coach_profile
            coach.preferred_preset = str(coach_state["preferred_preset"])
            coach.preset_affinities = dict(_decode(coach_state["preset_affinities"]))

    manager.dialogue_engine.market_step = int(snapshot["dialogue"]["market_step"])
    manager.dialogue_engine.topic_market = dict(
        _decode(snapshot["dialogue"]["topic_market"])
    )
    manager.narrative_event_bus.history = list(_decode(snapshot["narrative_history"]))
