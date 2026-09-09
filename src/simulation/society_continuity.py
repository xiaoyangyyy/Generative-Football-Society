"""Bounded cross-match continuity for SocietyAgent cognition and social state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict
from numbers import Integral, Real
from typing import Any, Mapping, TYPE_CHECKING

from src.simulation.psychological_state import PsychologicalState

if TYPE_CHECKING:
    from src.simulation.agent import SocietyAgent


SOCIETY_CONTINUITY_VERSION = 1
MAX_STATE_BYTES = 262_144
MAX_JSON_NODES = 8_192
MAX_STRING_LENGTH = 2_048
MAX_MAPPING_ITEMS = 128
MAX_SEQUENCE_ITEMS = 256

TACTICAL_BOUNDS = {
    "pressing_intensity": (0.0, 1.0),
    "risk_budget": (0.0, 1.0),
    "line_height": (0.0, 1.0),
    "rotation_aggressiveness": (0.0, 1.0),
}
APPRAISAL_BOUNDS = {
    "impact": (-1.0, 1.0),
    "novelty": (0.0, 1.0),
    "control": (0.0, 1.0),
    "certainty": (0.0, 1.0),
    "norm_violation": (0.0, 1.0),
    "agency": (-1.0, 1.0),
}
EMOTION_BOUNDS = {
    "pride": (0.0, 1.0),
    "anger": (0.0, 1.0),
    "shame": (0.0, 1.0),
    "fear": (0.0, 1.0),
    "determination": (0.0, 1.0),
}
COPING_BOUNDS = {
    "planning": (0.0, 1.0),
    "self_correction": (0.0, 1.0),
    "external_blame": (0.0, 1.0),
    "risk_shift": (-1.0, 1.0),
}
SOCIAL_BOUNDS = {
    "trust_index": (0.0, 1.0),
    "polarization": (0.0, 1.5),
    "narrative_fatigue": (0.0, 1.6),
}
PSYCHOLOGICAL_BOUNDS = {
    "morale": (-1.0, 1.0),
    "pressure": (-1.0, 1.0),
    "trust": (-1.0, 1.0),
    "risk_appetite": (-1.0, 1.0),
    "conflict": (-1.0, 1.0),
    "audience_activation": (-1.0, 1.0),
}
LATENT_BOUNDS = {
    "morale": (-30.0, 30.0),
    "stability": (-30.0, 30.0),
    "unity": (-30.0, 30.0),
    "confidence": (-30.0, 30.0),
    "risk_tolerance": (-30.0, 30.0),
    "pressing_intensity": (-30.0, 30.0),
    "referee_trust": (-30.0, 30.0),
    "media_sensitivity": (-30.0, 30.0),
}
SCALAR_BOUNDS = {
    "momentum": (-20.0, 20.0),
    "coach_authority": (0.0, 1.0),
    "icon_influence": (0.0, 1.0),
    "team_cohesion": (0.0, 1.0),
    "conflict_heat": (0.0, 1.05),
    "referee_trust": (0.0, 1.0),
    "referee_grievance": (0.0, 0.92),
    "icon_patience": (0.05, 1.2),
    "manager_pressure": (0.0, 1.0),
    "w_h": (0.15, 1.5),
    "w_x": (0.05, 1.5),
}
MEMORY_LIMITS = {
    "episodic_memory": 64,
    "procedural_memory": 32,
    "decision_memory": 32,
    "memory_event_log": 128,
    "beliefs": 16,
    "reflection_audit": 16,
}
STATE_KEYS = {
    "schema_version",
    "team_id",
    "source_transaction_id",
    "memory_clock",
    "scalars",
    "latent_state",
    "tactical_controls",
    "appraisal_state",
    "emotion_profile",
    "coping_profile",
    "social_narrative_state",
    "psychological_state",
    "semantic_memory",
    *MEMORY_LIMITS,
    "reflection_diary",
    "rivalry_database",
    "state_identity",
}
PUBLIC_BOUNDARY = (
    "simulator-owned social, psychological and cognitive continuity; memory text "
    "is not exposed here and no score or real-world causal claim is authorized"
)
META_PUBLIC_STATUSES = (
    "shadow",
    "observed_pending_evaluation",
    "committed",
    "rejected",
    "rolled_back",
    "expired",
)
META_PUBLIC_SCHEMA_VERSION = 2
META_PUBLIC_RECORD_VERSION = 1
META_PUBLIC_BOUNDARY = (
    "content-free simulator adaptation governance only; proposal text, memory "
    "evidence and matched rows are withheld, and outcome causality is not authorized"
)


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _meta_next_evidence(status: str) -> str:
    if status in {"shadow", "observed_pending_evaluation"}:
        return "identity_bound_matched_evaluation"
    if status == "committed":
        return "monitor_and_revalidate_before_reuse"
    return "new_identity_bound_proposal"


def _meta_authority_state(status: str) -> str:
    return {
        "committed": "granted",
        "rejected": "denied",
        "rolled_back": "rolled_back",
        "expired": "expired",
    }.get(status, "withheld")


def _meta_public_record(proposal: Any) -> dict[str, Any]:
    from src.simulation.meta_learning import PARAMETERS

    changes = [{
        "parameter": key,
        "time_scale": PARAMETERS[key].time_scale,
        "direction": (
            "increase" if change["delta"] > 0.0 else "decrease"
        ),
        "proposed_delta": float(change["delta"]),
    } for key, change in sorted(proposal.changes.items())]
    receipt = proposal.evaluation
    evaluation_available = bool(receipt)
    evaluation = {
        "available": evaluation_available,
        "evaluator": receipt.get("evaluator") if evaluation_available else None,
        "matched_units": (
            int(receipt["matched_units"]) if evaluation_available else 0
        ),
        "average_treatment_effect": (
            float(receipt["average_treatment_effect"])
            if evaluation_available else None
        ),
        "ci_low": float(receipt["ci_low"]) if evaluation_available else None,
        "ci_high": float(receipt["ci_high"]) if evaluation_available else None,
        "minimum_effect": (
            float(receipt["minimum_effect"])
            if evaluation_available else None
        ),
        "lower_bound_clears_threshold": bool(
            evaluation_available
            and receipt["ci_low"] > receipt["minimum_effect"]
        ),
        "receipt_identity": (
            receipt.get("receipt_identity") if evaluation_available else None
        ),
    }
    record = {
        "schema_version": META_PUBLIC_RECORD_VERSION,
        "proposal_id": proposal.proposal_id,
        "proposal_identity": proposal.proposal_identity,
        "status": proposal.status,
        "scope": proposal.scope,
        "observation_count": len(proposal.observations),
        "observation_window": proposal.expires_after,
        "changes": changes,
        "evaluation": evaluation,
        "authority_state": _meta_authority_state(proposal.status),
        "parameter_authority_granted": proposal.status == "committed",
        "next_required_evidence": _meta_next_evidence(proposal.status),
        "claim_boundary": META_PUBLIC_BOUNDARY,
    }
    record["record_identity"] = _identity(record)
    return record


def _validate_meta_public_record(payload: Mapping[str, Any]) -> None:
    from src.simulation.meta_learning import (
        MAX_PROPOSAL_OBSERVATIONS,
        MIN_MATCHED_EVALUATION_UNITS,
        MIN_SIMULATOR_UTILITY,
        MAX_SIMULATOR_UTILITY,
        PARAMETERS,
    )

    expected = {
        "schema_version", "proposal_id", "proposal_identity", "status",
        "scope", "observation_count", "observation_window", "changes",
        "evaluation", "authority_state", "parameter_authority_granted",
        "next_required_evidence", "claim_boundary", "record_identity",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("society public meta-learning record fields are invalid")
    frozen = copy.deepcopy(dict(payload))
    identity = frozen.pop("record_identity", None)
    proposal_id = payload.get("proposal_id")
    proposal_identity = payload.get("proposal_identity")
    status = payload.get("status")
    observation_count = payload.get("observation_count")
    observation_window = payload.get("observation_window")
    changes = payload.get("changes")
    if (
        payload.get("schema_version") != META_PUBLIC_RECORD_VERSION
        or not isinstance(proposal_id, str)
        or len(proposal_id) != 32
        or any(character not in "0123456789abcdef" for character in proposal_id)
        or not isinstance(proposal_identity, str)
        or len(proposal_identity) != 64
        or any(
            character not in "0123456789abcdef"
            for character in proposal_identity
        )
        or status not in META_PUBLIC_STATUSES
        or payload.get("scope") not in {"match", "tournament", "long_term"}
        or isinstance(observation_count, bool)
        or not isinstance(observation_count, int)
        or not 0 <= observation_count <= MAX_PROPOSAL_OBSERVATIONS
        or isinstance(observation_window, bool)
        or not isinstance(observation_window, int)
        or not 1 <= observation_window <= 128
        or observation_count > observation_window
        or not isinstance(changes, list)
        or not 1 <= len(changes) <= len(PARAMETERS)
        or not isinstance(identity, str)
        or identity != _identity(frozen)
        or payload.get("claim_boundary") != META_PUBLIC_BOUNDARY
    ):
        raise ValueError("society public meta-learning record is invalid")
    change_keys = []
    for change in changes:
        if not isinstance(change, Mapping) or set(change) != {
            "parameter", "time_scale", "direction", "proposed_delta",
        }:
            raise ValueError("society public meta-learning change is invalid")
        key = change.get("parameter")
        delta = change.get("proposed_delta")
        if (
            key not in PARAMETERS
            or change.get("time_scale") != PARAMETERS[key].time_scale
            or change.get("direction") not in {"increase", "decrease"}
            or isinstance(delta, bool)
            or not isinstance(delta, Real)
            or not math.isfinite(float(delta))
            or math.isclose(float(delta), 0.0, abs_tol=1e-12)
            or abs(float(delta)) > PARAMETERS[key].max_delta + 1e-12
            or (float(delta) > 0.0) != (change["direction"] == "increase")
        ):
            raise ValueError("society public meta-learning change is invalid")
        change_keys.append(key)
    if len(change_keys) != len(set(change_keys)) or change_keys != sorted(change_keys):
        raise ValueError("society public meta-learning changes are not canonical")
    evaluation = payload.get("evaluation")
    evaluation_fields = {
        "available", "evaluator", "matched_units",
        "average_treatment_effect", "ci_low", "ci_high",
        "minimum_effect", "lower_bound_clears_threshold",
        "receipt_identity",
    }
    if not isinstance(evaluation, Mapping) or set(evaluation) != evaluation_fields:
        raise ValueError("society public meta-learning evaluation is invalid")
    available = evaluation.get("available")
    if not isinstance(available, bool):
        raise ValueError("society public meta-learning evaluation is invalid")
    evaluated_status = status in {"committed", "rejected", "rolled_back"}
    if available != evaluated_status:
        raise ValueError("society public meta-learning evaluation state mismatch")
    if available:
        matched_units = evaluation.get("matched_units")
        values = [
            evaluation.get("average_treatment_effect"),
            evaluation.get("ci_low"), evaluation.get("ci_high"),
            evaluation.get("minimum_effect"),
        ]
        if (
            evaluation.get("evaluator") != "matched_seed_counterfactual_v1"
            or isinstance(matched_units, bool)
            or not isinstance(matched_units, int)
            or not MIN_MATCHED_EVALUATION_UNITS <= matched_units <= 128
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in values
            )
            or not float(evaluation["ci_low"])
            <= float(evaluation["average_treatment_effect"])
            <= float(evaluation["ci_high"])
            or not 0.0 <= float(evaluation["minimum_effect"]) <= (
                MAX_SIMULATOR_UTILITY - MIN_SIMULATOR_UTILITY
            )
            or not isinstance(
                evaluation.get("lower_bound_clears_threshold"), bool,
            )
            or evaluation["lower_bound_clears_threshold"]
            != (evaluation["ci_low"] > evaluation["minimum_effect"])
            or not isinstance(evaluation.get("receipt_identity"), str)
            or len(evaluation["receipt_identity"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in evaluation["receipt_identity"]
            )
        ):
            raise ValueError("society public meta-learning evaluation is invalid")
    elif (
        evaluation.get("evaluator") is not None
        or evaluation.get("matched_units") != 0
        or any(evaluation.get(field) is not None for field in (
            "average_treatment_effect", "ci_low", "ci_high",
            "minimum_effect", "receipt_identity",
        ))
        or evaluation.get("lower_bound_clears_threshold") is not False
    ):
        raise ValueError("empty society meta-learning evaluation is not empty")
    expected_authority = _meta_authority_state(status)
    if (
        payload.get("authority_state") != expected_authority
        or payload.get("parameter_authority_granted") is not (
            status == "committed"
        )
        or payload.get("next_required_evidence") != _meta_next_evidence(status)
        or status == "committed"
        and evaluation["lower_bound_clears_threshold"] is not True
        or status == "rejected"
        and evaluation["lower_bound_clears_threshold"] is not False
        or status == "rolled_back"
        and evaluation["lower_bound_clears_threshold"] is not True
    ):
        raise ValueError("society public meta-learning authority is invalid")


def _empty_meta_public_summary() -> dict[str, Any]:
    return {
        "schema_version": META_PUBLIC_SCHEMA_VERSION,
        "total": 0,
        **{status: 0 for status in META_PUBLIC_STATUSES},
        "records": [],
        "claim_boundary": META_PUBLIC_BOUNDARY,
    }


def _meta_public_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    from src.simulation.meta_learning import validate_meta_proposal

    counts = {status: 0 for status in META_PUBLIC_STATUSES}
    records = []
    for audit in state["reflection_audit"]:
        if "meta_proposal" not in audit:
            continue
        proposal = validate_meta_proposal(audit["meta_proposal"])
        if proposal.agent != state["team_id"]:
            raise ValueError("society meta-learning agent identity mismatch")
        counts[proposal.status] += 1
        records.append(_meta_public_record(proposal))
    records.sort(key=lambda row: row["proposal_id"])
    return {
        "schema_version": META_PUBLIC_SCHEMA_VERSION,
        "total": sum(counts.values()),
        **counts,
        "records": records,
        "claim_boundary": META_PUBLIC_BOUNDARY,
    }


def _validate_meta_public_summary(payload: Mapping[str, Any]) -> str:
    count_fields = {"total", *META_PUBLIC_STATUSES}
    governance_fields = count_fields | {
        "schema_version", "records", "claim_boundary",
    }
    if not isinstance(payload, Mapping) or frozenset(payload) not in {
        frozenset(count_fields), frozenset(governance_fields),
    }:
        raise ValueError("society public meta-learning summary is invalid")
    if any(
        isinstance(payload.get(field), bool)
        or not isinstance(payload.get(field), int)
        or not 0 <= payload[field] <= MEMORY_LIMITS["reflection_audit"]
        for field in count_fields
    ) or payload["total"] != sum(
        payload[status] for status in META_PUBLIC_STATUSES
    ):
        raise ValueError("society public meta-learning summary is invalid")
    if set(payload) == count_fields:
        return "counts"
    records = payload.get("records")
    if (
        payload.get("schema_version") != META_PUBLIC_SCHEMA_VERSION
        or payload.get("claim_boundary") != META_PUBLIC_BOUNDARY
        or not isinstance(records, list)
        or len(records) != payload["total"]
    ):
        raise ValueError("society public meta-learning governance is invalid")
    for record in records:
        _validate_meta_public_record(record)
    proposal_ids = [record["proposal_id"] for record in records]
    if (
        proposal_ids != sorted(proposal_ids)
        or len(proposal_ids) != len(set(proposal_ids))
        or any(
            payload[status]
            != sum(record["status"] == status for record in records)
            for status in META_PUBLIC_STATUSES
        )
    ):
        raise ValueError("society public meta-learning records are not canonical")
    return "governance"


def _meta_public_updates(
    before: Mapping[str, Any], after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    before_records = {
        row["proposal_id"]: row for row in before.get("records") or []
    }
    after_records = {
        row["proposal_id"]: row for row in after.get("records") or []
    }
    updates = []
    for proposal_id in sorted(set(before_records) | set(after_records)):
        left = before_records.get(proposal_id)
        right = after_records.get(proposal_id)
        if left == right:
            continue
        record = right or left
        if left is None:
            kind = "created"
        elif right is None:
            kind = "retention_removed"
        elif left["status"] != right["status"]:
            kind = "status_changed"
        elif left["observation_count"] != right["observation_count"]:
            kind = "observation_recorded"
        else:
            kind = "evidence_updated"
        update = {
            "proposal_id": proposal_id,
            "proposal_identity": record["proposal_identity"],
            "update_kind": kind,
            "before_status": left["status"] if left else None,
            "after_status": right["status"] if right else None,
            "observation_count_before": (
                left["observation_count"] if left else 0
            ),
            "observation_count_after": (
                right["observation_count"] if right else 0
            ),
            "observation_count_delta": (
                (right["observation_count"] if right else 0)
                - (left["observation_count"] if left else 0)
            ),
            "observation_window": record["observation_window"],
            "changes": copy.deepcopy(record["changes"]),
            "evaluation_after": (
                copy.deepcopy(right["evaluation"]) if right else None
            ),
            "authority_state": (
                right["authority_state"] if right else "not_retained"
            ),
            "next_required_evidence": (
                right["next_required_evidence"]
                if right else "record_not_retained"
            ),
            "claim_boundary": META_PUBLIC_BOUNDARY,
        }
        update["update_identity"] = _identity(update)
        updates.append(update)
    return updates


def _bounded_json(value: Any, *, nodes: list[int], depth: int = 0) -> Any:
    nodes[0] += 1
    if nodes[0] > MAX_JSON_NODES or depth > 12:
        raise ValueError("society continuity JSON exceeds structural limits")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        result = int(value)
        if abs(result) > 1_000_000_000:
            raise ValueError("society continuity integer is out of bounds")
        return result
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result) or abs(result) > 1_000_000_000.0:
            raise ValueError("society continuity number is invalid")
        return result
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise ValueError("society continuity string exceeds limit")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise ValueError("society continuity mapping exceeds limit")
        if any(
            not isinstance(key, str) or not key or len(key) > 128
            for key in value
        ):
            raise ValueError("society continuity mapping key is invalid")
        output = {}
        for key in sorted(value):
            output[key] = _bounded_json(
                value[key], nodes=nodes, depth=depth + 1,
            )
        return output
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise ValueError("society continuity sequence exceeds limit")
        return [
            _bounded_json(item, nodes=nodes, depth=depth + 1)
            for item in value
        ]
    raise ValueError(
        f"unsupported society continuity value: {type(value).__name__}"
    )


def _number_map(
    value: Any, bounds: Mapping[str, tuple[float, float]], *, label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(bounds):
        raise ValueError(f"society continuity {label} fields are invalid")
    result = {}
    for key, (low, high) in bounds.items():
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(f"society continuity {label} value is invalid")
        number = float(raw)
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"society continuity {label} value is out of bounds")
        result[key] = number
    return result


def _bounded_records(value: Any, *, field: str) -> list[dict[str, Any]]:
    limit = MEMORY_LIMITS[field]
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"society continuity {field} exceeds limit")
    if any(not isinstance(record, Mapping) for record in value):
        raise ValueError(f"society continuity {field} record is invalid")
    records = [copy.deepcopy(dict(record)) for record in value]
    if field in {"episodic_memory", "procedural_memory"}:
        ids = [record.get("id") for record in records]
        if any(
            not isinstance(item, str) or not item or len(item) > 160
            for item in ids
        ) or len(ids) != len(set(ids)):
            raise ValueError(f"society continuity {field} identities are invalid")
    return records


def validate_society_continuity_state(
    payload: Mapping[str, Any], *, expected_team: str | None = None,
) -> dict[str, Any]:
    """Validate, normalize and identity-check one persisted society state."""
    if not isinstance(payload, Mapping) or set(payload) != STATE_KEYS:
        raise ValueError("society continuity state fields are invalid")
    canonical = _bounded_json(dict(payload), nodes=[0])
    identity = canonical.pop("state_identity", None)
    if (
        canonical.get("schema_version") != SOCIETY_CONTINUITY_VERSION
        or not isinstance(canonical.get("team_id"), str)
        or not canonical["team_id"]
        or len(canonical["team_id"]) > 160
        or expected_team is not None
        and canonical["team_id"] != expected_team
        or not isinstance(canonical.get("source_transaction_id"), str)
        or not canonical["source_transaction_id"]
        or len(canonical["source_transaction_id"]) > 256
        or not isinstance(identity, str)
        or len(identity) != 64
        or identity != _identity(canonical)
    ):
        raise ValueError("society continuity state identity mismatch")
    clock = canonical.get("memory_clock")
    if (
        isinstance(clock, bool)
        or not isinstance(clock, int)
        or not 0 <= clock <= 10_000_000
    ):
        raise ValueError("society continuity memory clock is invalid")
    canonical["scalars"] = _number_map(
        canonical["scalars"], SCALAR_BOUNDS, label="scalars",
    )
    canonical["latent_state"] = _number_map(
        canonical["latent_state"], LATENT_BOUNDS, label="latent state",
    )
    canonical["tactical_controls"] = _number_map(
        canonical["tactical_controls"], TACTICAL_BOUNDS,
        label="tactical controls",
    )
    canonical["appraisal_state"] = _number_map(
        canonical["appraisal_state"], APPRAISAL_BOUNDS,
        label="appraisal state",
    )
    canonical["emotion_profile"] = _number_map(
        canonical["emotion_profile"], EMOTION_BOUNDS,
        label="emotion profile",
    )
    canonical["coping_profile"] = _number_map(
        canonical["coping_profile"], COPING_BOUNDS,
        label="coping profile",
    )
    canonical["social_narrative_state"] = _number_map(
        canonical["social_narrative_state"], SOCIAL_BOUNDS,
        label="social narrative state",
    )
    canonical["psychological_state"] = _number_map(
        canonical["psychological_state"], PSYCHOLOGICAL_BOUNDS,
        label="psychological state",
    )
    if not isinstance(canonical["semantic_memory"], Mapping):
        raise ValueError("society continuity semantic memory is invalid")
    for field in MEMORY_LIMITS:
        canonical[field] = _bounded_records(canonical[field], field=field)
    from src.simulation.meta_learning import validate_meta_proposal

    for audit in canonical["reflection_audit"]:
        if "meta_proposal" not in audit:
            continue
        proposal = validate_meta_proposal(audit["meta_proposal"])
        if proposal.agent != canonical["team_id"]:
            raise ValueError("society meta-learning agent identity mismatch")
    if not isinstance(canonical["reflection_diary"], str):
        raise ValueError("society continuity reflection diary is invalid")
    rivalry = canonical["rivalry_database"]
    if (
        not isinstance(rivalry, Mapping)
        or len(rivalry) > 64
        or any(
            not isinstance(team, str)
            or not team
            or len(team) > 160
            or isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not -10.0 <= float(value) <= 10.0
            for team, value in rivalry.items()
        )
    ):
        raise ValueError("society continuity rivalry database is invalid")
    canonical["rivalry_database"] = {
        key: float(value) for key, value in rivalry.items()
    }
    canonical["state_identity"] = identity
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_STATE_BYTES:
        raise ValueError("society continuity state exceeds byte limit")
    return canonical


def _compact_memory_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(record))
    output["content"] = str(output.get("content") or "")[:500]
    output["summary"] = str(
        output.get("summary") or output["content"]
    )[:500]
    tags = output.get("tags")
    tags = tags if isinstance(tags, (list, tuple)) else []
    output["tags"] = [str(value)[:80] for value in tags[:16]]
    metadata = output.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    output["metadata"] = {
        key: copy.deepcopy(metadata[key])
        for key in ("continuity_event_id", "applied")
        if key in metadata
    }
    provenance = output.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    parent_ids = provenance.get("causal_parent_ids")
    parent_ids = parent_ids if isinstance(parent_ids, (list, tuple)) else []
    contradicts = provenance.get("contradicts")
    contradicts = contradicts if isinstance(contradicts, (list, tuple)) else []
    output["provenance"] = {
        "source": str(provenance.get("source") or "simulation")[:80],
        "causal_parent_ids": [
            str(value)[:160] for value in parent_ids[-16:]
        ],
        "contradicts": [
            str(value)[:160] for value in contradicts[-16:]
        ],
        "valid_from": provenance.get("valid_from"),
        "valid_until": provenance.get("valid_until"),
    }
    return output


def _compact_decision_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(record[key])
        for key in (
            "step", "opponent", "stage", "opponent_style",
            "stage_pressure", "controls", "outcomes", "evidence_id",
        )
        if key in record
    }


def _compact_event_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(record[key])
        for key in ("step", "layer", "p_write", "id")
        if key in record
    }


def _recent(records: Any, field: str) -> list[Any]:
    if not isinstance(records, list):
        return []
    selected = records[-MEMORY_LIMITS[field]:]
    if field in {"episodic_memory", "procedural_memory"}:
        return [
            _compact_memory_record(record)
            for record in selected if isinstance(record, Mapping)
        ]
    if field == "decision_memory":
        return [
            _compact_decision_record(record)
            for record in selected if isinstance(record, Mapping)
        ]
    if field == "memory_event_log":
        return [
            _compact_event_record(record)
            for record in selected if isinstance(record, Mapping)
        ]
    return copy.deepcopy(selected)


def capture_society_continuity(
    agent: "SocietyAgent", *, source_transaction_id: str,
) -> dict[str, Any]:
    """Capture the bounded state that can alter later social/cognitive behavior."""
    psychological = getattr(agent, "psychological_state", PsychologicalState())
    if not isinstance(psychological, PsychologicalState):
        raise ValueError("SocietyAgent psychological state is invalid")
    roles = getattr(agent, "roles", {})
    icon = roles.get("Icon", {}) if isinstance(roles, Mapping) else {}
    manager = roles.get("Manager", {}) if isinstance(roles, Mapping) else {}
    payload = {
        "schema_version": SOCIETY_CONTINUITY_VERSION,
        "team_id": str(agent.team_name),
        "source_transaction_id": str(source_transaction_id),
        "memory_clock": int(getattr(agent, "memory_clock", 0)),
        "scalars": {
            "momentum": float(agent.momentum),
            "coach_authority": float(agent.coach_authority),
            "icon_influence": float(agent.icon_influence),
            "team_cohesion": float(agent.team_cohesion),
            "conflict_heat": float(agent.conflict_heat),
            "referee_trust": float(agent.referee_trust),
            "referee_grievance": float(agent.referee_grievance),
            "icon_patience": float(icon.get("patience", 0.8)),
            "manager_pressure": float(manager.get("pressure", 0.0)),
            "w_h": float(agent.W_h),
            "w_x": float(agent.W_x),
        },
        "latent_state": copy.deepcopy(agent.z_state),
        "tactical_controls": copy.deepcopy(agent.tactical_controls),
        "appraisal_state": copy.deepcopy(agent.appraisal_state),
        "emotion_profile": copy.deepcopy(agent.emotion_profile),
        "coping_profile": copy.deepcopy(agent.coping_profile),
        "social_narrative_state": copy.deepcopy(agent.social_narrative_state),
        "psychological_state": asdict(psychological),
        "semantic_memory": copy.deepcopy(agent.semantic_memory),
        "episodic_memory": _recent(agent.episodic_memory, "episodic_memory"),
        "procedural_memory": _recent(
            agent.procedural_memory, "procedural_memory",
        ),
        "decision_memory": _recent(agent.decision_memory, "decision_memory"),
        "memory_event_log": _recent(
            agent.memory_event_log, "memory_event_log",
        ),
        "beliefs": _recent(agent.beliefs, "beliefs"),
        "reflection_audit": _recent(
            agent.llm_reflection_audit, "reflection_audit",
        ),
        "reflection_diary": str(agent.reflection_diary)[:MAX_STRING_LENGTH],
        "rivalry_database": copy.deepcopy(agent.rivalry_database),
    }
    canonical = _bounded_json(payload, nodes=[0])
    canonical["state_identity"] = _identity(canonical)
    return validate_society_continuity_state(
        canonical, expected_team=str(agent.team_name),
    )


def apply_society_continuity(
    agent: "SocietyAgent", payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore one validated state onto a freshly constructed SocietyAgent."""
    state = validate_society_continuity_state(
        payload, expected_team=str(agent.team_name),
    )
    scalars = state["scalars"]
    agent.momentum = scalars["momentum"]
    agent.coach_authority = scalars["coach_authority"]
    agent.icon_influence = scalars["icon_influence"]
    agent.team_cohesion = scalars["team_cohesion"]
    agent.conflict_heat = scalars["conflict_heat"]
    agent.referee_trust = scalars["referee_trust"]
    agent.referee_grievance = scalars["referee_grievance"]
    agent.W_h = scalars["w_h"]
    agent.W_x = scalars["w_x"]
    agent.roles["Icon"]["patience"] = scalars["icon_patience"]
    agent.roles["Manager"]["pressure"] = scalars["manager_pressure"]
    agent.z_state = copy.deepcopy(state["latent_state"])
    agent.set_tactical_controls(copy.deepcopy(state["tactical_controls"]))
    agent.appraisal_state = copy.deepcopy(state["appraisal_state"])
    agent.emotion_profile = copy.deepcopy(state["emotion_profile"])
    agent.coping_profile = copy.deepcopy(state["coping_profile"])
    agent.social_narrative_state = copy.deepcopy(
        state["social_narrative_state"]
    )
    agent.psychological_state = PsychologicalState(
        **state["psychological_state"]
    )
    agent.semantic_memory = copy.deepcopy(state["semantic_memory"])
    agent.episodic_memory = copy.deepcopy(state["episodic_memory"])
    agent.procedural_memory = copy.deepcopy(state["procedural_memory"])
    agent.decision_memory = copy.deepcopy(state["decision_memory"])
    agent.memory_event_log = copy.deepcopy(state["memory_event_log"])
    agent.beliefs = copy.deepcopy(state["beliefs"])
    agent.llm_reflection_audit = copy.deepcopy(state["reflection_audit"])
    agent.reflection_diary = state["reflection_diary"]
    agent.rivalry_database = copy.deepcopy(state["rivalry_database"])
    agent.memory_clock = state["memory_clock"]
    return state


def build_society_decision_context(
    agent: "SocietyAgent",
) -> dict[str, Any] | None:
    """Build a prompt-safe, bounded prior-world packet for the next coach trigger."""
    carry = getattr(agent, "squad_carryover", None)
    source = getattr(carry, "society_state", None)
    if not isinstance(source, Mapping):
        return None
    state = validate_society_continuity_state(
        source, expected_team=str(agent.team_name),
    )
    psychological = getattr(agent, "psychological_state", PsychologicalState())
    if not isinstance(psychological, PsychologicalState):
        raise ValueError("SocietyAgent psychological state is invalid")
    candidates = [
        record
        for record in agent.episodic_memory + agent.procedural_memory
        if isinstance(record, Mapping)
    ]
    candidates.sort(
        key=lambda record: (
            float(record.get("salience", 0.0)),
            int(record.get("created_step", 0)),
        ),
        reverse=True,
    )
    memories = [{
        "id": str(record.get("id", ""))[:160],
        "event": str(record.get("event", ""))[:80],
        "summary": str(
            record.get("summary") or record.get("content") or ""
        )[:200],
        "salience": round(float(record.get("salience", 0.0)), 6),
        "stage": str(record.get("stage") or "")[:80],
        "opponent": str(record.get("opponent") or "")[:160],
    } for record in candidates[:4]]
    beliefs = sorted(
        (
            belief for belief in agent.beliefs
            if isinstance(belief, Mapping)
        ),
        key=lambda belief: float(belief.get("confidence", 0.0)),
        reverse=True,
    )
    recent = agent.decision_memory[-1] if agent.decision_memory else None
    context = {
        "schema_version": SOCIETY_CONTINUITY_VERSION,
        "source_state_identity": state["state_identity"],
        "source_transaction_id": state["source_transaction_id"],
        "psychological_decision_modifiers": psychological.decision_modifiers(),
        "social_narrative_state": copy.deepcopy(agent.social_narrative_state),
        "emotion_profile": copy.deepcopy(agent.emotion_profile),
        "referee_grievance": float(agent.referee_grievance),
        "tactical_controls": copy.deepcopy(agent.tactical_controls),
        "high_salience_memories": memories,
        "beliefs": [{
            "id": str(belief.get("id", ""))[:160],
            "claim": str(belief.get("claim", ""))[:200],
            "confidence": round(float(belief.get("confidence", 0.0)), 6),
        } for belief in beliefs[:3]],
        "recent_decision": copy.deepcopy(recent) if recent else None,
        "memory_text_authority": "untrusted_context_only",
        "claim_boundary": (
            "prior simulated social and cognitive state may inform a bounded "
            "plan but cannot override current match facts, validators or score"
        ),
    }
    canonical = _bounded_json(context, nodes=[0])
    canonical["context_identity"] = _identity(canonical)
    return canonical


def society_public_snapshot(
    payload: Mapping[str, Any] | None, *, team_id: str,
) -> dict[str, Any]:
    """Return a content-free product projection of persisted society state."""
    if payload is None:
        return {
            "available": False,
            "state_identity": None,
            "source_transaction_id": None,
            "memory_records": 0,
            "cognitive_memory_records": 0,
            "beliefs": 0,
            "reflections": 0,
            "tactical_controls": {},
            "emotion_profile": {},
            "social_narrative_state": {},
            "psychological_state": {},
            "referee_grievance": None,
            "meta_learning": _empty_meta_public_summary(),
            "claim_boundary": PUBLIC_BOUNDARY,
        }
    state = validate_society_continuity_state(
        payload, expected_team=team_id,
    )
    memories = state["episodic_memory"] + state["procedural_memory"]
    return {
        "available": True,
        "state_identity": state["state_identity"],
        "source_transaction_id": state["source_transaction_id"],
        "memory_records": len(memories),
        "cognitive_memory_records": sum(
            record.get("event") == "micro_cognitive" for record in memories
        ),
        "beliefs": len(state["beliefs"]),
        "reflections": len(state["reflection_audit"]),
        "tactical_controls": copy.deepcopy(state["tactical_controls"]),
        "emotion_profile": copy.deepcopy(state["emotion_profile"]),
        "social_narrative_state": copy.deepcopy(
            state["social_narrative_state"]
        ),
        "psychological_state": copy.deepcopy(state["psychological_state"]),
        "referee_grievance": state["scalars"]["referee_grievance"],
        "meta_learning": _meta_public_summary(state),
        "claim_boundary": PUBLIC_BOUNDARY,
    }


def validate_society_public_snapshot(payload: Mapping[str, Any]) -> None:
    """Validate the content-free projection embedded in product evidence."""
    legacy_expected = {
        "available", "state_identity", "source_transaction_id",
        "memory_records", "cognitive_memory_records", "beliefs",
        "reflections", "tactical_controls", "emotion_profile",
        "social_narrative_state", "psychological_state",
        "referee_grievance", "claim_boundary",
    }
    expected = legacy_expected | {"meta_learning"}
    if (
        not isinstance(payload, Mapping)
        or frozenset(payload)
        not in {frozenset(legacy_expected), frozenset(expected)}
        or not isinstance(payload.get("available"), bool)
        or payload.get("claim_boundary") != PUBLIC_BOUNDARY
    ):
        raise ValueError("society public snapshot fields are invalid")
    for field in (
        "memory_records", "cognitive_memory_records", "beliefs", "reflections",
    ):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("society public snapshot count is invalid")
    if (
        payload["memory_records"]
        > MEMORY_LIMITS["episodic_memory"] + MEMORY_LIMITS["procedural_memory"]
        or payload["beliefs"] > MEMORY_LIMITS["beliefs"]
        or payload["reflections"] > MEMORY_LIMITS["reflection_audit"]
    ):
        raise ValueError("society public snapshot count exceeds limit")
    if payload["cognitive_memory_records"] > payload["memory_records"]:
        raise ValueError("society public cognitive count is invalid")
    meta = payload.get("meta_learning")
    if meta is not None:
        _validate_meta_public_summary(meta)
    if payload["available"] is False:
        if (
            payload["state_identity"] is not None
            or payload["source_transaction_id"] is not None
            or any(payload[field] != 0 for field in (
                "memory_records", "cognitive_memory_records", "beliefs",
                "reflections",
            ))
            or any(payload[field] != {} for field in (
                "tactical_controls", "emotion_profile",
                "social_narrative_state", "psychological_state",
            ))
            or payload["referee_grievance"] is not None
            or meta is not None and (
                meta.get("total") != 0 or bool(meta.get("records"))
            )
        ):
            raise ValueError("unavailable society public snapshot is not empty")
        return
    identity = payload["state_identity"]
    source = payload["source_transaction_id"]
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
        or not isinstance(source, str)
        or not source
        or len(source) > 256
    ):
        raise ValueError("society public snapshot identity is invalid")
    _number_map(
        payload["tactical_controls"], TACTICAL_BOUNDS,
        label="public tactical controls",
    )
    _number_map(
        payload["emotion_profile"], EMOTION_BOUNDS,
        label="public emotion profile",
    )
    _number_map(
        payload["social_narrative_state"], SOCIAL_BOUNDS,
        label="public social narrative state",
    )
    _number_map(
        payload["psychological_state"], PSYCHOLOGICAL_BOUNDS,
        label="public psychological state",
    )
    grievance = payload["referee_grievance"]
    if (
        isinstance(grievance, bool)
        or not isinstance(grievance, Real)
        or not 0.0 <= float(grievance) <= 0.92
    ):
        raise ValueError("society public referee grievance is invalid")


def _public_state_changes(
    before: Mapping[str, Any], after: Mapping[str, Any],
) -> list[dict[str, Any]]:
    changes = []
    for scope in (
        "tactical_controls", "emotion_profile", "social_narrative_state",
        "psychological_state",
    ):
        left = before.get(scope)
        right = after.get(scope)
        left = left if isinstance(left, Mapping) else {}
        right = right if isinstance(right, Mapping) else {}
        for field in sorted(set(left) | set(right)):
            if left.get(field) != right.get(field):
                changes.append({
                    "scope": scope,
                    "field": field,
                    "before": left.get(field),
                    "after": right.get(field),
                })
    if before.get("referee_grievance") != after.get("referee_grievance"):
        changes.append({
            "scope": "referee_grievance",
            "field": "value",
            "before": before.get("referee_grievance"),
            "after": after.get("referee_grievance"),
        })
    return changes


def society_public_transition(
    before: Mapping[str, Any], after: Mapping[str, Any], *,
    include_state_changes: bool = False,
) -> dict[str, Any]:
    """Describe direct persisted changes without assigning outcome causality."""
    validate_society_public_snapshot(before)
    validate_society_public_snapshot(after)
    before_available = before.get("available") is True
    after_available = after.get("available") is True
    available = before_available or after_available
    if not available:
        return {
            "available": False,
            "reason": "society_continuity_not_recorded",
            "claim_boundary": PUBLIC_BOUNDARY,
        }
    changed = []
    for field in (
        "tactical_controls", "emotion_profile", "social_narrative_state",
        "psychological_state", "referee_grievance",
    ):
        if before.get(field) != after.get(field):
            changed.append(field)
    transition = {
        "available": True,
        "before_available": before_available,
        "after_available": after_available,
        "before_state_identity": before.get("state_identity"),
        "after_state_identity": after.get("state_identity"),
        "source_transaction_id": after.get("source_transaction_id"),
        "memory_record_delta": int(after.get("memory_records") or 0)
        - int(before.get("memory_records") or 0),
        "cognitive_memory_delta": int(
            after.get("cognitive_memory_records") or 0
        ) - int(before.get("cognitive_memory_records") or 0),
        "belief_delta": int(after.get("beliefs") or 0)
        - int(before.get("beliefs") or 0),
        "reflection_delta": int(after.get("reflections") or 0)
        - int(before.get("reflections") or 0),
        "changed_state_fields": changed,
        "claim_boundary": PUBLIC_BOUNDARY,
        **(
            {
                "meta_learning_before": copy.deepcopy(
                    before.get("meta_learning") or {
                        "total": 0,
                        **{status: 0 for status in META_PUBLIC_STATUSES},
                    }
                ),
                "meta_learning_after": copy.deepcopy(
                    after.get("meta_learning") or {
                        "total": 0,
                        **{status: 0 for status in META_PUBLIC_STATUSES},
                    }
                ),
                "meta_learning_delta": {
                    field: int((after.get("meta_learning") or {}).get(field, 0))
                    - int((before.get("meta_learning") or {}).get(field, 0))
                    for field in ("total", *META_PUBLIC_STATUSES)
                },
                **(
                    {
                        "meta_learning_updates": _meta_public_updates(
                            before.get("meta_learning") or {},
                            after.get("meta_learning") or {},
                        )
                    }
                    if "records" in (before.get("meta_learning") or {})
                    or "records" in (after.get("meta_learning") or {})
                    else {}
                ),
            }
            if "meta_learning" in before or "meta_learning" in after else {}
        ),
    }
    if include_state_changes:
        transition["state_changes"] = _public_state_changes(before, after)
    return transition


__all__ = [
    "SOCIETY_CONTINUITY_VERSION",
    "apply_society_continuity",
    "build_society_decision_context",
    "capture_society_continuity",
    "society_public_snapshot",
    "society_public_transition",
    "validate_society_public_snapshot",
    "validate_society_continuity_state",
]
