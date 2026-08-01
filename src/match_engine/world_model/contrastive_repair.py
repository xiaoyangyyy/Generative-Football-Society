"""One-shot LLM explanation repair driven by world-model counterevidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import numpy as np

from src.match_engine.world_model.contrastive_explanation import (
    CONTEXT_FACTORS,
    evaluate_llm_contrastive_claim,
    validate_llm_contrastive_claim,
)


CONTRASTIVE_REPAIR_VERSION = 1


def llm_contrastive_repair_signature(model_name: str) -> str:
    payload = json.dumps({
        "contract_version": CONTRASTIVE_REPAIR_VERSION,
        "model": str(model_name or "rule_fallback"),
        "task": "one_shot_world_model_counterevidence_explanation_repair",
    }, sort_keys=True, separators=(",", ":"))
    return "llm-contrastive-repair:" + hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def _revision_contract(packet: dict[str, Any]) -> dict[str, Any]:
    actions = []
    horizons_by_action = {}
    for candidate in packet.get("candidates") or []:
        action = str(candidate.get("action", "")).lower()
        if not action:
            continue
        actions.append(action)
        horizons_by_action[action] = sorted(
            str(key)
            for key, prediction in (
                candidate.get("multi_horizon_predictions") or {}
            ).items()
            if int((prediction or {}).get("rollout_steps", 1)) in {1, 2}
        )
    return {
        "evaluated_actions": sorted(set(actions)),
        "validated_horizons_by_action": horizons_by_action,
        "allowed_factors": list(CONTEXT_FACTORS),
        "selected_action_is_immutable": True,
        "revision_calls_remaining": 1,
    }


def attempt_contrastive_explanation_repair(
    llm,
    runtime,
    state,
    packet: dict[str, Any],
    initial_audit: dict[str, Any],
    *,
    team_id: str,
    selected_action: str,
    enabled: bool,
    repair_signature: str = "contrastive-repair-contract-unspecified",
    total_member_trajectory_path_budget: int = 128,
) -> dict[str, Any]:
    """Give one model-counterevidence turn without reopening action choice."""
    selected = str(selected_action).lower()
    total_budget = max(
        0, min(128, int(total_member_trajectory_path_budget)),
    )
    try:
        initial_paths = max(0, int(initial_audit.get(
            "trajectory_member_paths", 0,
        )))
    except (TypeError, ValueError, OverflowError):
        initial_paths = total_budget + 1
    base = {
        "version": CONTRASTIVE_REPAIR_VERSION,
        "enabled": bool(enabled),
        "repair_attempted": False,
        "revision_accepted": False,
        "repair_successful": False,
        "llm_calls": 0,
        "llm_call_budget": 1,
        "initial_member_trajectory_paths": initial_paths,
        "total_member_trajectory_paths": initial_paths,
        "total_member_trajectory_path_budget": total_budget,
        "selected_action": selected,
        "selected_action_immutable": True,
        "action_mutated": False,
        "control_fields_mutated": False,
        "world_model_prediction_mutated": False,
        "shadow_only": True,
        "authority_active": False,
        "causal_interpretation": False,
        "can_change_selected_action": False,
        "can_update_world_model": False,
        "repair_signature": str(repair_signature),
        "checkpoint_signature": str(initial_audit.get(
            "checkpoint_signature", "runtime_unspecified",
        )),
        "environment_signature": str(initial_audit.get(
            "environment_signature", "environment_unspecified",
        )),
    }
    if not enabled:
        return {**base, "reason": "contrastive_repair_disabled"}
    if not initial_audit.get("accepted"):
        return {**base, "reason": "initial_contrastive_claim_not_accepted"}
    if initial_audit.get("directionally_faithful"):
        return {**base, "reason": "initial_explanation_already_faithful"}
    initial_claim = validate_llm_contrastive_claim(
        initial_audit.get("claim")
    )
    if initial_claim is None or initial_claim["selected_action"] != selected:
        return {**base, "reason": "initial_repair_contract_invalid"}
    revision_provider = getattr(
        llm, "revise_world_model_contrastive_claim", None,
    )
    if not callable(revision_provider):
        return {**base, "reason": "llm_has_no_contrastive_repair_interface"}
    remaining_paths = total_budget - initial_paths
    if remaining_paths <= 0:
        return {**base, "reason": "contrastive_repair_budget_exhausted"}
    contract = _revision_contract(packet)
    counterevidence = {
        "initial_claim": initial_claim,
        "observed_margin": initial_audit.get(
            "observed_selected_vs_alternative_margin"
        ),
        "neutralized_margin": initial_audit.get(
            "neutralized_selected_vs_alternative_margin"
        ),
        "declared_factor_effect_on_margin": initial_audit.get(
            "declared_factor_effect_on_margin"
        ),
        "claimed_directional_effect": initial_audit.get(
            "claimed_directional_effect"
        ),
        "minimum_directional_effect": initial_audit.get(
            "minimum_directional_effect"
        ),
        "directionally_faithful": False,
        "neutral_reference": initial_audit.get("neutral_reference"),
        "context_intervention_linf": initial_audit.get(
            "context_intervention_linf"
        ),
    }
    try:
        raw_revision = revision_provider(
            team_name=str(team_id),
            immutable_selected_action=selected,
            counterevidence=counterevidence,
            revision_contract=contract,
        )
        if isinstance(raw_revision, str):
            raw_revision = json.loads(raw_revision)
    except (RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return {
            **base,
            "repair_attempted": True,
            "llm_calls": 1,
            "reason": "contrastive_repair_llm_call_failed",
            "counterevidence": counterevidence,
            "revision_contract": contract,
        }
    revised_claim = validate_llm_contrastive_claim(raw_revision)
    attempted = {
        **base,
        "repair_attempted": True,
        "llm_calls": 1,
        "counterevidence": counterevidence,
        "revision_contract": contract,
        "raw_revision_valid": revised_claim is not None,
    }
    if revised_claim is None:
        return {
            **attempted,
            "reason": "contrastive_repair_revision_invalid",
        }
    if revised_claim["selected_action"] != selected:
        return {
            **attempted,
            "reason": "contrastive_repair_attempted_action_change",
            "revised_claim": revised_claim,
            "selected_action_immutable": False,
        }
    revised_audit = evaluate_llm_contrastive_claim(
        runtime,
        state,
        packet,
        revised_claim,
        team_id=str(team_id),
        selected_action=selected,
        contrastive_signature=str(initial_audit.get(
            "contrastive_signature", "contrastive-contract-unspecified",
        )),
        max_member_trajectory_paths=min(64, remaining_paths),
    )
    revised_paths = max(0, int(revised_audit.get(
        "trajectory_member_paths", 0,
    )))
    total_paths = initial_paths + max(0, revised_paths)
    initial_directional = float(initial_audit.get(
        "claimed_directional_effect", 0.0,
    ))
    revised_directional = float(revised_audit.get(
        "claimed_directional_effect", 0.0,
    ))
    revision_accepted = bool(revised_audit.get("accepted"))
    successful = bool(
        revision_accepted
        and revised_audit.get("directionally_faithful")
        and revised_directional > initial_directional
        and total_paths <= total_budget
    )
    return {
        **attempted,
        "reason": (
            "contrastive_explanation_repaired"
            if successful else "contrastive_repair_not_improved"
        ),
        "revision_accepted": revision_accepted,
        "repair_successful": successful,
        "revised_claim": revised_claim,
        "revised_audit": revised_audit,
        "initial_directional_effect": initial_directional,
        "revised_directional_effect": revised_directional,
        "directional_effect_gain": revised_directional - initial_directional,
        "revised_member_trajectory_paths": revised_paths,
        "total_member_trajectory_paths": total_paths,
        "effective_claim": revised_claim if successful else initial_claim,
        "effective_audit_source": "revision" if successful else "initial",
    }


def contrastive_repair_diagnostics(
    match_logs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    match_rows = []
    for payload in match_logs:
        rows = []
        for record in (
            (payload.get("world_model_decision_adoption") or {}).get("records")
            or []
        ):
            audit = record.get("llm_contrastive_repair_context")
            if isinstance(audit, dict) and audit.get("repair_attempted"):
                rows.append(audit)
        if rows:
            match_rows.append(rows)
    valid = []
    malformed = 0
    for rows in match_rows:
        for row in rows:
            try:
                version = int(row["version"])
                calls = int(row["llm_calls"])
                call_budget = int(row["llm_call_budget"])
                paths = int(row["total_member_trajectory_paths"])
                path_budget = int(row["total_member_trajectory_path_budget"])
                initial_directional = float(row.get(
                    "initial_directional_effect", 0.0,
                ))
                revised_directional = float(row.get(
                    "revised_directional_effect", 0.0,
                ))
            except (KeyError, TypeError, ValueError, OverflowError):
                malformed += 1
                continue
            revised = row.get("revised_audit") or {}
            revised_claim = validate_llm_contrastive_claim(
                row.get("revised_claim")
            )
            revision_contract_valid = bool(
                not row.get("revision_accepted")
                or (
                    revised.get("accepted")
                    and revised_claim is not None
                    and revised_claim["selected_action"]
                    == str(row.get("selected_action", ""))
                )
            )
            expected_success = bool(
                row.get("revision_accepted")
                and revised.get("directionally_faithful")
                and revised_directional > initial_directional
                and paths <= path_budget
            )
            if (
                version != CONTRASTIVE_REPAIR_VERSION
                or calls != 1 or call_budget != 1
                or paths < 0 or path_budget <= 0 or paths > path_budget
                or not np.isfinite(initial_directional)
                or not np.isfinite(revised_directional)
                or bool(row.get("repair_successful")) != expected_success
                or not row.get("selected_action_immutable")
                or not revision_contract_valid
            ):
                malformed += 1
                continue
            valid.append(row)
    valid_ids = {id(row) for row in valid}
    valid_match_rows = [
        [row for row in rows if id(row) in valid_ids]
        for rows in match_rows
    ]
    valid_match_rows = [rows for rows in valid_match_rows if rows]
    match_success = [
        float(np.mean([
            float(bool(row.get("repair_successful"))) for row in rows
        ]))
        for rows in valid_match_rows
    ]
    match_gains = [
        float(np.mean([
            float(row.get("directional_effect_gain", 0.0)) for row in rows
        ]))
        for rows in valid_match_rows
    ]
    signatures = sorted({
        str(row.get("repair_signature", "")) for row in valid
    })
    scopes = sorted({(
        str(row.get("checkpoint_signature", "")),
        str(row.get("environment_signature", "")),
    ) for row in valid})
    unspecified = {
        "", "contrastive-repair-contract-unspecified",
        "runtime_unspecified", "environment_unspecified",
    }
    return {
        "version": CONTRASTIVE_REPAIR_VERSION,
        "evaluation_kind": "one_shot_contrastive_explanation_repair",
        "attempts": len(valid),
        "matches": len(valid_match_rows),
        "malformed_repair_audits": malformed,
        "revisions_accepted": sum(
            bool(row.get("revision_accepted")) for row in valid
        ),
        "repairs_successful": sum(
            bool(row.get("repair_successful")) for row in valid
        ),
        "match_clustered_repair_success_rate": (
            float(np.mean(match_success)) if match_success else 0.0
        ),
        "match_clustered_directional_effect_gain": (
            float(np.mean(match_gains)) if match_gains else 0.0
        ),
        "all_single_call": all(
            int(row.get("llm_calls", 0)) == 1 for row in valid
        ),
        "budgets_respected": all(
            int(row.get("total_member_trajectory_paths", 0))
            <= int(row.get("total_member_trajectory_path_budget", 0))
            for row in valid
        ),
        "all_actions_immutable": all(
            bool(row.get("selected_action_immutable"))
            and not bool(row.get("action_mutated"))
            and not bool(row.get("control_fields_mutated"))
            and not bool(row.get("can_change_selected_action"))
            for row in valid
        ),
        "all_shadow_only": all(bool(row.get("shadow_only")) for row in valid),
        "all_non_controlling": all(
            not bool(row.get("authority_active"))
            and not bool(row.get("world_model_prediction_mutated"))
            for row in valid
        ),
        "all_non_causal": all(
            not bool(row.get("causal_interpretation")) for row in valid
        ),
        "provenance_compatible": bool(
            len(signatures) == 1 and signatures[0] not in unspecified
            and len(scopes) == 1
            and all(value not in unspecified for value in scopes[0])
        ),
        "repair_signatures": signatures,
        "model_policy_scopes": [list(scope) for scope in scopes],
    }
