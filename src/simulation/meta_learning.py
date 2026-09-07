"""Audited, time-scale-aware application of LLM reflection proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import uuid
from typing import Any, Mapping

import numpy as np


META_PROPOSAL_VERSION = 2
META_EVALUATION_VERSION = 1
META_OBSERVATION_VERSION = 1
MIN_MATCHED_EVALUATION_UNITS = 8
MAX_PROPOSAL_OBSERVATIONS = 16
MIN_SIMULATOR_UTILITY = -5.0
MAX_SIMULATOR_UTILITY = 5.0
ALLOWED_STATUSES = {
    "shadow",
    "observed_pending_evaluation",
    "committed",
    "rejected",
    "rolled_back",
    "expired",
}


class _NoActionableMetaAdjustment(ValueError):
    """Signal that reflection produced no parameter-changing proposal."""


@dataclass(frozen=True)
class MetaParameter:
    time_scale: str
    max_delta: float
    low: float
    high: float


PARAMETERS = {
    "risk_budget": MetaParameter("fast", 0.10, 0.0, 1.0),
    "pressing_intensity": MetaParameter("fast", 0.10, 0.0, 1.0),
    "line_height": MetaParameter("fast", 0.10, 0.0, 1.0),
    "rotation_aggressiveness": MetaParameter("fast", 0.10, 0.0, 1.0),
    "icon_patience": MetaParameter("medium", 0.06, 0.05, 1.2),
    "w_h_delta": MetaParameter("slow", 0.025, 0.15, 1.5),
    "w_x_delta": MetaParameter("slow", 0.025, 0.05, 1.5),
}


@dataclass
class MetaProposal:
    proposal_id: str
    agent: str
    reflection: str
    evidence_ids: tuple[str, ...]
    changes: dict[str, dict[str, float]]
    status: str = "shadow"
    evaluation: dict[str, Any] = field(default_factory=dict)
    scope: str = "match"
    expires_after: int = 1
    created_step: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    proposal_identity: str = ""
    authorization_identity: str | None = None


def _canonical_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"Meta-learning {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Meta-learning {label} must be finite")
    return result


def _bounded_int(value: Any, *, low: int, high: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Meta-learning {label} must be an integer")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"Meta-learning {label} is out of bounds")
    return result


def _read_parameter(agent: Any, key: str) -> float:
    if key in agent.tactical_controls:
        return float(agent.tactical_controls[key])
    if key == "icon_patience":
        return float(agent.roles["Icon"]["patience"])
    return float(agent.W_h if key == "w_h_delta" else agent.W_x)


def _write_parameter(agent: Any, key: str, value: float) -> None:
    if key in agent.tactical_controls:
        agent.tactical_controls[key] = value
    elif key == "icon_patience":
        agent.roles["Icon"]["patience"] = value
    elif key == "w_h_delta":
        agent.W_h = value
    else:
        agent.W_x = value


def _proposal_core(proposal: MetaProposal) -> dict[str, Any]:
    return {
        "schema_version": META_PROPOSAL_VERSION,
        "proposal_id": proposal.proposal_id,
        "agent": proposal.agent,
        "reflection": proposal.reflection,
        "evidence_ids": list(proposal.evidence_ids),
        "changes": proposal.changes,
        "scope": proposal.scope,
        "expires_after": proposal.expires_after,
        "created_step": proposal.created_step,
    }


def meta_proposal_to_dict(proposal: MetaProposal) -> dict[str, Any]:
    return {
        **_proposal_core(proposal),
        "status": proposal.status,
        "evaluation": proposal.evaluation,
        "observations": proposal.observations,
        "proposal_identity": proposal.proposal_identity,
        "authorization_identity": proposal.authorization_identity,
    }


def _validate_observation(
    payload: Mapping[str, Any], *, proposal_identity: str,
) -> dict[str, Any]:
    expected = {
        "schema_version", "operation_id", "proposal_identity",
        "result_utility", "goal_difference", "xg_difference",
        "observation_identity", "claim_boundary",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Meta-learning observation fields are invalid")
    canonical = dict(payload)
    identity = canonical.pop("observation_identity", None)
    operation_id = canonical.get("operation_id")
    for key in ("result_utility", "goal_difference", "xg_difference"):
        canonical[key] = _finite(
            canonical[key], label=f"observation {key}",
        )
    if not (
        MIN_SIMULATOR_UTILITY
        <= canonical["result_utility"]
        <= MAX_SIMULATOR_UTILITY
    ):
        raise ValueError("Meta-learning observation utility is out of bounds")
    if (
        canonical.get("schema_version") != META_OBSERVATION_VERSION
        or not isinstance(operation_id, str)
        or not operation_id
        or len(operation_id) > 256
        or canonical.get("proposal_identity") != proposal_identity
        or canonical.get("claim_boundary")
        != "descriptive_post_proposal_world_observation_not_effect_evidence"
        or not _is_sha256(identity)
        or identity != _canonical_identity(canonical)
    ):
        raise ValueError("Meta-learning observation identity mismatch")
    canonical["observation_identity"] = identity
    return canonical


def validate_meta_proposal(payload: Mapping[str, Any]) -> MetaProposal:
    expected = {
        "schema_version", "proposal_id", "agent", "reflection",
        "evidence_ids", "changes", "status", "evaluation", "scope",
        "expires_after", "created_step", "observations",
        "proposal_identity", "authorization_identity",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Meta-learning proposal fields are invalid")
    proposal_id = payload.get("proposal_id")
    agent = payload.get("agent")
    reflection = payload.get("reflection")
    evidence_ids = payload.get("evidence_ids")
    changes = payload.get("changes")
    status = payload.get("status")
    scope = payload.get("scope")
    evaluation = payload.get("evaluation")
    observations = payload.get("observations")
    if (
        payload.get("schema_version") != META_PROPOSAL_VERSION
        or not isinstance(proposal_id, str)
        or not proposal_id
        or len(proposal_id) > 128
        or not isinstance(agent, str)
        or not agent
        or len(agent) > 160
        or not isinstance(reflection, str)
        or len(reflection) > 2_048
        or not isinstance(evidence_ids, list)
        or len(evidence_ids) > 32
        or any(
            not isinstance(value, str) or not value or len(value) > 160
            for value in evidence_ids
        )
        or len(evidence_ids) != len(set(evidence_ids))
        or not isinstance(changes, Mapping)
        or not changes
        or len(changes) > len(PARAMETERS)
        or any(key not in PARAMETERS for key in changes)
        or status not in ALLOWED_STATUSES
        or scope not in {"match", "tournament", "long_term"}
        or not isinstance(evaluation, Mapping)
        or not isinstance(observations, list)
        or len(observations) > MAX_PROPOSAL_OBSERVATIONS
    ):
        raise ValueError("Meta-learning proposal content is invalid")
    normalized_changes: dict[str, dict[str, float]] = {}
    for key, raw in changes.items():
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"before", "after", "delta"}
        ):
            raise ValueError("Meta-learning change fields are invalid")
        spec = PARAMETERS[key]
        before = _finite(raw["before"], label=f"{key} before")
        after = _finite(raw["after"], label=f"{key} after")
        delta = _finite(raw["delta"], label=f"{key} delta")
        if (
            not spec.low <= before <= spec.high
            or not spec.low <= after <= spec.high
            or abs(delta) > spec.max_delta + 1e-12
            or not math.isclose(after - before, delta, abs_tol=1e-12)
        ):
            raise ValueError("Meta-learning change is out of bounds")
        normalized_changes[key] = {
            "before": before, "after": after, "delta": delta,
        }
    expires_after = _bounded_int(
        payload.get("expires_after"), low=1, high=128, label="expiry",
    )
    created_step = _bounded_int(
        payload.get("created_step"), low=0, high=10_000_000,
        label="created step",
    )
    proposal = MetaProposal(
        proposal_id=proposal_id,
        agent=agent,
        reflection=reflection,
        evidence_ids=tuple(evidence_ids),
        changes=normalized_changes,
        status=status,
        evaluation=dict(evaluation),
        scope=scope,
        expires_after=expires_after,
        created_step=created_step,
        observations=[],
        proposal_identity=str(payload.get("proposal_identity") or ""),
        authorization_identity=payload.get("authorization_identity"),
    )
    expected_identity = _canonical_identity(_proposal_core(proposal))
    if (
        not _is_sha256(proposal.proposal_identity)
        or proposal.proposal_identity != expected_identity
        or proposal.authorization_identity is not None
        and not _is_sha256(proposal.authorization_identity)
    ):
        raise ValueError("Meta-learning proposal identity mismatch")
    proposal.observations = [
        _validate_observation(
            observation, proposal_identity=proposal.proposal_identity,
        )
        for observation in observations
    ]
    identities = [
        row["observation_identity"] for row in proposal.observations
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Meta-learning observations are duplicated")
    if status in {"committed", "rejected", "rolled_back"}:
        validate_meta_evaluation_receipt(
            evaluation,
            proposal_identity=proposal.proposal_identity,
            agent=proposal.agent,
        )
        if proposal.authorization_identity != evaluation["receipt_identity"]:
            raise ValueError("Meta-learning authorization identity mismatch")
    elif evaluation or proposal.authorization_identity is not None:
        raise ValueError("Unevaluated meta-learning proposal has authorization")
    return proposal


def _normalize_matched_rows(rows: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(rows, (list, tuple))
        or not MIN_MATCHED_EVALUATION_UNITS <= len(rows) <= 128
    ):
        raise ValueError("Meta-learning matched rows are out of bounds")
    normalized = []
    for row in rows:
        expected = {
            "unit_id", "seed", "control_utility", "treated_utility",
            "control_source_identity", "treated_source_identity",
        }
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError("Meta-learning matched row fields are invalid")
        unit_id = row.get("unit_id")
        seed = row.get("seed")
        control_identity = row.get("control_source_identity")
        treated_identity = row.get("treated_source_identity")
        if (
            not isinstance(unit_id, str)
            or not unit_id
            or len(unit_id) > 160
            or isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or not 0 <= int(seed) <= 2**63 - 1
            or not _is_sha256(control_identity)
            or not _is_sha256(treated_identity)
            or control_identity == treated_identity
        ):
            raise ValueError("Meta-learning matched row identity is invalid")
        control_utility = _finite(
            row["control_utility"], label="control utility",
        )
        treated_utility = _finite(
            row["treated_utility"], label="treated utility",
        )
        if not (
            MIN_SIMULATOR_UTILITY <= control_utility <= MAX_SIMULATOR_UTILITY
            and MIN_SIMULATOR_UTILITY
            <= treated_utility
            <= MAX_SIMULATOR_UTILITY
        ):
            raise ValueError("Meta-learning matched utility is out of bounds")
        normalized.append({
            "unit_id": unit_id,
            "seed": int(seed),
            "control_utility": control_utility,
            "treated_utility": treated_utility,
            "control_source_identity": control_identity,
            "treated_source_identity": treated_identity,
        })
    unit_ids = [row["unit_id"] for row in normalized]
    seeds = [row["seed"] for row in normalized]
    if len(unit_ids) != len(set(unit_ids)) or len(seeds) != len(set(seeds)):
        raise ValueError("Meta-learning matched row identities are duplicated")
    return sorted(normalized, key=lambda row: (row["unit_id"], row["seed"]))


def _evaluation_statistics(
    rows: list[dict[str, Any]],
) -> tuple[float, float, float]:
    effects = np.asarray([
        row["treated_utility"] - row["control_utility"] for row in rows
    ], dtype=float)
    ate = float(np.mean(effects))
    standard_error = (
        float(np.std(effects, ddof=1) / math.sqrt(len(effects)))
        if len(effects) > 1 else 0.0
    )
    radius = 1.96 * standard_error
    return ate, ate - radius, ate + radius


def build_meta_evaluation_receipt(
    proposal: MetaProposal,
    *,
    matched_rows: list[Mapping[str, Any]],
    minimum_effect: float = 0.0,
    uses_sealed_data: bool = False,
) -> dict[str, Any]:
    validate_meta_proposal(meta_proposal_to_dict(proposal))
    rows = _normalize_matched_rows(matched_rows)
    ate, ci_low, ci_high = _evaluation_statistics(rows)
    normalized_minimum = _finite(
        minimum_effect, label="minimum effect",
    )
    source_identity = _canonical_identity({
        "metric_contract": "simulator_outcome_utility_v1",
        "matched_rows": rows,
    })
    payload = {
        "schema_version": META_EVALUATION_VERSION,
        "evaluator": "matched_seed_counterfactual_v1",
        "metric_contract": "simulator_outcome_utility_v1",
        "proposal_identity": proposal.proposal_identity,
        "agent": proposal.agent,
        "source_evidence_identity": source_identity,
        "matched_units": len(rows),
        "matched_rows": rows,
        "interval_method": "matched_unit_normal_95_v1",
        "average_treatment_effect": ate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "minimum_effect": normalized_minimum,
        "uses_sealed_data": uses_sealed_data,
        "claim_boundary": (
            "simulator_matched_seed_meta_parameter_effect_only_no_real_world_claim"
        ),
    }
    receipt = {**payload, "receipt_identity": _canonical_identity(payload)}
    return validate_meta_evaluation_receipt(
        receipt,
        proposal_identity=proposal.proposal_identity,
        agent=proposal.agent,
    )


def validate_meta_evaluation_receipt(
    payload: Mapping[str, Any], *, proposal_identity: str, agent: str,
) -> dict[str, Any]:
    expected = {
        "schema_version", "evaluator", "metric_contract",
        "proposal_identity", "agent", "source_evidence_identity",
        "matched_units", "matched_rows", "interval_method",
        "average_treatment_effect", "ci_low", "ci_high",
        "minimum_effect", "uses_sealed_data", "claim_boundary",
        "receipt_identity",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Meta-learning evaluation receipt fields are invalid")
    canonical = dict(payload)
    receipt_identity = canonical.pop("receipt_identity", None)
    rows = _normalize_matched_rows(canonical.get("matched_rows"))
    replay_ate, replay_low, replay_high = _evaluation_statistics(rows)
    ate = _finite(
        canonical.get("average_treatment_effect"), label="evaluation ATE",
    )
    ci_low = _finite(canonical.get("ci_low"), label="evaluation CI low")
    ci_high = _finite(canonical.get("ci_high"), label="evaluation CI high")
    minimum = _finite(
        canonical.get("minimum_effect"), label="minimum effect",
    )
    matched_units = _bounded_int(
        canonical.get("matched_units"),
        low=MIN_MATCHED_EVALUATION_UNITS,
        high=10_000,
        label="matched units",
    )
    canonical.update({
        "matched_units": matched_units,
        "matched_rows": rows,
        "average_treatment_effect": ate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "minimum_effect": minimum,
    })
    if (
        canonical.get("schema_version") != META_EVALUATION_VERSION
        or canonical.get("evaluator") != "matched_seed_counterfactual_v1"
        or canonical.get("metric_contract") != "simulator_outcome_utility_v1"
        or canonical.get("interval_method") != "matched_unit_normal_95_v1"
        or canonical.get("proposal_identity") != proposal_identity
        or canonical.get("agent") != agent
        or canonical.get("source_evidence_identity")
        != _canonical_identity({
            "metric_contract": "simulator_outcome_utility_v1",
            "matched_rows": rows,
        })
        or matched_units != len(rows)
        or not 0.0 <= minimum <= (
            MAX_SIMULATOR_UTILITY - MIN_SIMULATOR_UTILITY
        )
        or canonical.get("uses_sealed_data") is not False
        or canonical.get("claim_boundary")
        != "simulator_matched_seed_meta_parameter_effect_only_no_real_world_claim"
        or not math.isclose(ate, replay_ate, abs_tol=1e-12)
        or not math.isclose(ci_low, replay_low, abs_tol=1e-12)
        or not math.isclose(ci_high, replay_high, abs_tol=1e-12)
        or not _is_sha256(receipt_identity)
        or receipt_identity != _canonical_identity(canonical)
    ):
        raise ValueError("Meta-learning evaluation receipt is invalid")
    canonical["receipt_identity"] = receipt_identity
    return canonical


class MetaLearningController:
    def __init__(self, audit_path: str | Path | None = None) -> None:
        self.proposals: dict[str, MetaProposal] = {}
        self.audit_path = Path(audit_path) if audit_path else None

    def _audit(self, proposal: MetaProposal, action: str) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "proposal_id": proposal.proposal_id,
            "proposal_identity": proposal.proposal_identity,
            "agent": proposal.agent,
            "action": action,
            "status": proposal.status,
            "scope": proposal.scope,
            "expires_after": proposal.expires_after,
            "evidence_ids": list(proposal.evidence_ids),
            "changes": proposal.changes,
            "evaluation": proposal.evaluation,
            "observation_count": len(proposal.observations),
        }
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def propose(
        self, agent: Any, reflection: dict[str, Any] | None, *,
        operation_id: str | None = None,
    ) -> MetaProposal:
        reflection = reflection or {}
        if (
            operation_id is not None
            and (
                not isinstance(operation_id, str)
                or not operation_id
                or len(operation_id) > 256
            )
        ):
            raise ValueError("Invalid meta-adaptation operation identity")
        if reflection.get("uses_sealed_data"):
            raise ValueError("sealed data cannot be used for meta-adaptation")
        suggestion = dict(reflection.get("suggested_adjustments") or {})
        if any(
            not isinstance(key, str) or not key or len(key) > 128
            for key in suggestion
        ):
            raise ValueError("Invalid meta-adaptation parameter identity")
        legacy = {
            "new_wh": ("w_h_delta", agent.W_h),
            "new_wx": ("w_x_delta", agent.W_x),
            "new_icon_patience": ("icon_patience", agent.roles["Icon"]["patience"]),
        }
        for source, (target, current) in legacy.items():
            if source in reflection:
                if (
                    isinstance(reflection[source], bool)
                    or not isinstance(reflection[source], Real)
                ):
                    continue
                try:
                    suggestion[target] = float(reflection[source]) - float(current)
                except (TypeError, ValueError):
                    pass
        known = {
            str(record.get("id"))
            for record in getattr(agent, "memory_event_log", [])
            if isinstance(record, Mapping) and record.get("id")
        }
        raw_evidence = reflection.get("evidence_memory_ids") or []
        if not isinstance(raw_evidence, (list, tuple)):
            raise ValueError("Meta-learning evidence IDs must be a sequence")
        requested = tuple(dict.fromkeys(
            str(value) for value in raw_evidence[:32] if str(value)
        ))
        verified = tuple(value for value in requested if value in known)
        raw_confidence = reflection.get("confidence", 0.5)
        try:
            confidence = (
                float(raw_confidence)
                if not isinstance(raw_confidence, bool)
                and isinstance(raw_confidence, Real)
                else 0.5
            )
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = float(np.clip(
            confidence if math.isfinite(confidence) else 0.5, 0.0, 1.0,
        ))
        evidence_factor = 1.0 if verified else 0.65
        changes = {}
        for key, raw_delta in suggestion.items():
            spec = PARAMETERS.get(key)
            if spec is None:
                continue
            if isinstance(raw_delta, bool) or not isinstance(raw_delta, Real):
                continue
            try:
                proposed = float(raw_delta)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(proposed):
                continue
            before = _read_parameter(agent, key)
            bounded = spec.max_delta * float(np.tanh(proposed / spec.max_delta))
            after = float(np.clip(
                before + bounded * confidence * evidence_factor,
                spec.low,
                spec.high,
            ))
            if math.isclose(after, before, abs_tol=1e-12):
                continue
            changes[key] = {"before": before, "after": after, "delta": after - before}
        if not changes:
            raise _NoActionableMetaAdjustment(
                "Reflection produced no actionable meta-adjustment"
            )
        proposal_id = (
            hashlib.sha256(
                f"reflection:{agent.name}:{operation_id}".encode("utf-8")
            ).hexdigest()[:32]
            if operation_id else uuid.uuid4().hex
        )
        expires_after = _bounded_int(
            reflection.get("expires_after", 1),
            low=1,
            high=128,
            label="expiry",
        )
        created_step = _bounded_int(
            reflection.get("created_step", 0),
            low=0,
            high=10_000_000,
            label="created step",
        )
        proposal = MetaProposal(
            proposal_id=proposal_id,
            agent=str(agent.name),
            reflection=str(
                reflection.get("reflection", reflection.get("diary", ""))
            )[:2_048],
            evidence_ids=verified,
            changes=changes,
            scope=str(reflection.get("scope", "match")),
            expires_after=expires_after,
            created_step=created_step,
        )
        proposal.proposal_identity = _canonical_identity(
            _proposal_core(proposal)
        )
        proposal = validate_meta_proposal(meta_proposal_to_dict(proposal))
        existing = self.proposals.get(proposal.proposal_id)
        if existing is not None:
            if existing.proposal_identity == proposal.proposal_identity:
                return existing
            raise ValueError("Meta-learning operation identity was reused")
        self.proposals[proposal.proposal_id] = proposal
        self._audit(proposal, "proposed_shadow")
        return proposal

    def observe(
        self,
        proposal: MetaProposal,
        *,
        operation_id: str,
        result_utility: float,
        goal_difference: float,
        xg_difference: float,
    ) -> MetaProposal:
        proposal = validate_meta_proposal(meta_proposal_to_dict(proposal))
        if proposal.status not in {"shadow", "observed_pending_evaluation"}:
            return proposal
        if any(
            row["operation_id"] == operation_id
            for row in proposal.observations
        ):
            return proposal
        if len(proposal.observations) >= min(
            proposal.expires_after, MAX_PROPOSAL_OBSERVATIONS,
        ):
            proposal.status = "expired"
            self.proposals[proposal.proposal_id] = proposal
            self._audit(proposal, "expired_without_evaluation")
            return proposal
        normalized_result = _finite(
            result_utility, label="observation result_utility",
        )
        normalized_goal_difference = _finite(
            goal_difference, label="observation goal_difference",
        )
        normalized_xg_difference = _finite(
            xg_difference, label="observation xg_difference",
        )
        core = {
            "schema_version": META_OBSERVATION_VERSION,
            "operation_id": operation_id,
            "proposal_identity": proposal.proposal_identity,
            "result_utility": normalized_result,
            "goal_difference": normalized_goal_difference,
            "xg_difference": normalized_xg_difference,
            "claim_boundary": (
                "descriptive_post_proposal_world_observation_not_effect_evidence"
            ),
        }
        observation = _validate_observation(
            {**core, "observation_identity": _canonical_identity(core)},
            proposal_identity=proposal.proposal_identity,
        )
        proposal.observations.append(observation)
        proposal.status = "observed_pending_evaluation"
        self.proposals[proposal.proposal_id] = proposal
        self._audit(proposal, "observed_without_effect_authority")
        return proposal

    def evaluate_and_commit(
        self,
        agent: Any,
        proposal: MetaProposal,
        evaluation_receipt: Mapping[str, Any],
    ) -> MetaProposal:
        proposal = validate_meta_proposal(meta_proposal_to_dict(proposal))
        receipt = validate_meta_evaluation_receipt(
            evaluation_receipt,
            proposal_identity=proposal.proposal_identity,
            agent=proposal.agent,
        )
        if proposal.agent != str(agent.name):
            raise ValueError("Meta-learning proposal agent mismatch")
        if proposal.status == "committed":
            if proposal.authorization_identity == receipt["receipt_identity"]:
                return proposal
            raise ValueError(
                "Meta-learning proposal already has other authority"
            )
        if proposal.status not in {"shadow", "observed_pending_evaluation"}:
            raise ValueError(f"Proposal is not evaluable: {proposal.status}")
        proposal.evaluation = receipt
        proposal.authorization_identity = receipt["receipt_identity"]
        if receipt["ci_low"] <= receipt["minimum_effect"]:
            proposal.status = "rejected"
            self.proposals[proposal.proposal_id] = proposal
            self._audit(proposal, "rejected_by_matched_evaluation")
            return proposal
        for key, change in proposal.changes.items():
            if not np.isclose(_read_parameter(agent, key), change["before"]):
                raise RuntimeError(f"Parameter changed since proposal: {key}")
        for key, change in proposal.changes.items():
            _write_parameter(agent, key, change["after"])
        proposal.status = "committed"
        self.proposals[proposal.proposal_id] = proposal
        self._audit(proposal, "committed_by_matched_evaluation")
        return proposal

    def rollback(self, agent: Any, proposal: MetaProposal) -> MetaProposal:
        proposal = validate_meta_proposal(meta_proposal_to_dict(proposal))
        if proposal.status != "committed":
            raise ValueError(f"Proposal is not rollbackable: {proposal.status}")
        for key, change in proposal.changes.items():
            if not np.isclose(_read_parameter(agent, key), change["after"]):
                raise RuntimeError(f"Parameter changed after proposal: {key}")
        for key, change in proposal.changes.items():
            _write_parameter(agent, key, change["before"])
        proposal.status = "rolled_back"
        self.proposals[proposal.proposal_id] = proposal
        self._audit(proposal, "rolled_back")
        return proposal

    def expire(self, agent: Any, *, current_step: int) -> list[str]:
        expired = []
        for proposal_id, proposal in list(self.proposals.items()):
            if current_step < proposal.created_step + proposal.expires_after:
                continue
            if proposal.status == "committed":
                proposal = self.rollback(agent, proposal)
            elif proposal.status in {"shadow", "observed_pending_evaluation"}:
                proposal.status = "expired"
                self._audit(proposal, "expired_without_authority")
            else:
                continue
            self.proposals[proposal_id] = proposal
            expired.append(proposal_id)
        return expired

    def apply(
        self, agent: Any, reflection: dict[str, Any] | None, *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """Stage a shadow proposal; never grant parameter authority directly."""
        reflection = reflection or {}
        diary = str(
            reflection.get("reflection", reflection.get("diary", ""))
        )[:2_048]
        agent.reflection_diary = diary
        raw_evidence = reflection.get("evidence_memory_ids") or []
        requested_evidence = (
            [str(value) for value in raw_evidence[:32]]
            if isinstance(raw_evidence, (list, tuple)) else []
        )
        raw_audit_confidence = reflection.get("confidence", 0.5)
        try:
            audit_confidence = (
                float(raw_audit_confidence)
                if not isinstance(raw_audit_confidence, bool)
                and isinstance(raw_audit_confidence, Real)
                else 0.5
            )
        except (TypeError, ValueError):
            audit_confidence = 0.5
        audit_confidence = float(np.clip(
            audit_confidence if math.isfinite(audit_confidence) else 0.5,
            0.0,
            1.0,
        ))
        try:
            proposal = self.propose(
                agent, reflection, operation_id=operation_id,
            )
        except _NoActionableMetaAdjustment:
            suggestion = dict(reflection.get("suggested_adjustments") or {})
            rejected = {}
            for key, value in suggestion.items():
                if key not in PARAMETERS:
                    rejected[key] = "unknown_parameter"
                    continue
                rejected[key] = (
                    "no_effective_delta"
                    if not isinstance(value, bool)
                    and isinstance(value, Real)
                    and math.isfinite(float(value))
                    else "non_numeric_delta"
                )
            return {
                "agent": str(agent.name),
                "operation_id": operation_id,
                "reflection": diary,
                "proposal_id": None,
                "proposal_identity": None,
                "proposal_status": "not_proposed",
                "confidence": audit_confidence,
                "evidence_requested": requested_evidence,
                "evidence_verified": [],
                "evidence_factor": 0.0,
                "proposed_adjustments": {},
                "applied_adjustments": {},
                "rejected_adjustments": rejected,
                "authorization_required": False,
                "claim_boundary": (
                    "reflection_recorded_without_actionable_meta_adjustment"
                ),
            }
        proposed = {
            key: {
                "time_scale": PARAMETERS[key].time_scale,
                "proposed_delta": change["delta"],
                "before": change["before"],
                "after": change["after"],
            }
            for key, change in proposal.changes.items()
        }
        suggestion = dict(reflection.get("suggested_adjustments") or {})
        rejected = {
            key: (
                "unknown_parameter"
                if key not in PARAMETERS else "non_numeric_delta"
            )
            for key, value in suggestion.items()
            if key not in proposal.changes
            and (
                key not in PARAMETERS
                or isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            )
        }
        confidence = audit_confidence
        return {
            "agent": str(agent.name),
            "operation_id": operation_id,
            "reflection": diary,
            "proposal_id": proposal.proposal_id,
            "proposal_identity": proposal.proposal_identity,
            "proposal_status": proposal.status,
            "confidence": confidence,
            "evidence_requested": requested_evidence,
            "evidence_verified": list(proposal.evidence_ids),
            "evidence_factor": 1.0 if proposal.evidence_ids else 0.65,
            "proposed_adjustments": proposed,
            "applied_adjustments": {},
            "rejected_adjustments": rejected,
            "authorization_required": True,
            "meta_proposal": meta_proposal_to_dict(proposal),
            "claim_boundary": (
                "shadow_meta_proposal_only_until_identity_bound_matched_evaluation"
            ),
        }

    def authorize_from_agent_audit(
        self,
        agent: Any,
        proposal_id: str,
        evaluation_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        matches = [
            record
            for record in getattr(agent, "llm_reflection_audit", [])
            if isinstance(record, dict)
            and record.get("proposal_id") == proposal_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "Meta-learning proposal audit identity is ambiguous"
            )
        audit = matches[0]
        proposal = validate_meta_proposal(
            audit.get("meta_proposal") or {}
        )
        committed = self.evaluate_and_commit(
            agent, proposal, evaluation_receipt,
        )
        audit["proposal_status"] = committed.status
        audit["meta_proposal"] = meta_proposal_to_dict(committed)
        audit["authorization_identity"] = committed.authorization_identity
        audit["applied_adjustments"] = (
            committed.changes if committed.status == "committed" else {}
        )
        return audit


def observe_agent_meta_proposals(
    agent: Any,
    *,
    operation_id: str,
    result_utility: float,
    goal_difference: float,
    xg_difference: float,
) -> list[str]:
    """Attach idempotent descriptive outcomes to persisted shadow proposals."""
    observed: list[str] = []
    controller = MetaLearningController()
    for audit in getattr(agent, "llm_reflection_audit", []):
        if not isinstance(audit, dict) or "meta_proposal" not in audit:
            continue
        proposal = validate_meta_proposal(audit["meta_proposal"])
        if proposal.agent != str(agent.name):
            raise ValueError("Meta-learning audit agent mismatch")
        before = len(proposal.observations)
        proposal = controller.observe(
            proposal,
            operation_id=operation_id,
            result_utility=result_utility,
            goal_difference=goal_difference,
            xg_difference=xg_difference,
        )
        audit["proposal_status"] = proposal.status
        audit["meta_proposal"] = meta_proposal_to_dict(proposal)
        audit["observation_count"] = len(proposal.observations)
        if len(proposal.observations) > before:
            observed.append(proposal.proposal_id)
    return observed


__all__ = [
    "META_PROPOSAL_VERSION",
    "MIN_MATCHED_EVALUATION_UNITS",
    "MetaLearningController",
    "MetaParameter",
    "MetaProposal",
    "PARAMETERS",
    "build_meta_evaluation_receipt",
    "meta_proposal_to_dict",
    "observe_agent_meta_proposals",
    "validate_meta_evaluation_receipt",
    "validate_meta_proposal",
]
