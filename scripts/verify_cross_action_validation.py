#!/usr/bin/env python3
"""Read-only verification of the cross-action planning authority contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.match_engine.world_model.cross_validation import (  # noqa: E402
    CROSS_MIN_GROUPS,
    CROSS_MIN_SAMPLES,
    CROSS_MIN_SKILL,
    replay_cross_action_validation,
)
from src.match_engine.world_model.config import WorldModelConfig  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path = ROOT) -> dict:
    protocol_path = root / "data/evaluation/cross_action_validation_protocol_v1.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validation = protocol.get("validation") or {}
    execution = protocol.get("current_execution") or {}
    expected_contract = {
        "scope": "grouped_heldout_simulator_cross_transitions",
        "data_domain": "simulator_transition_rows_only",
        "external_football_validity": False,
        "baseline": "same_state_persistence",
        "metric": "weighted_next_observation_mse",
        "minimum_samples": CROSS_MIN_SAMPLES,
        "minimum_groups": CROSS_MIN_GROUPS,
        "minimum_skill_vs_persistence": CROSS_MIN_SKILL,
        "runtime_minimum_planner_quality": WorldModelConfig().min_planner_quality,
        "quality_is_recomputed_at_load": True,
        "tampered_or_incomplete_evidence_fails_closed": True,
    }
    contract_verified = bool(
        protocol.get("schema_version") == 1
        and protocol.get("protocol_id") == "gfs-cross-action-planning-authority-v1"
        and protocol.get("state") == "registered_validation_contract"
        and protocol.get("action") == "cross"
        and validation == expected_contract
    )
    required = [str(value) for value in protocol.get("required_code_paths") or []]
    code_paths_verified = bool(required) and all((root / value).is_file() for value in required)
    candidate = protocol.get("candidate") or {}
    checkpoint_path = (root / str(candidate.get("checkpoint") or "")).resolve()
    checkpoint_path.relative_to(root.resolve())
    checkpoint_identity_verified = bool(
        checkpoint_path.is_file()
        and _sha256(checkpoint_path) == candidate.get("checkpoint_sha256")
    )
    checkpoint_validation = {
        "valid": False, "quality": 0.0,
        "reason": "checkpoint_identity_mismatch",
    }
    if checkpoint_identity_verified:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = payload.get("meta") if isinstance(payload, dict) else {}
        model_validation = (
            metadata.get("validation") if isinstance(metadata, dict) else {}
        )
        checkpoint_validation = replay_cross_action_validation(
            model_validation.get("cross_action_validation")
            if isinstance(model_validation, dict) else None
        )
    authority = bool(
        contract_verified and code_paths_verified and checkpoint_identity_verified
        and checkpoint_validation.get("valid")
        and checkpoint_validation.get("quality", 0.0)
        >= expected_contract["runtime_minimum_planner_quality"]
    )
    status = (
        "authorized_cross_planning" if authority
        else "blocked_checkpoint_missing_cross_validation"
        if checkpoint_identity_verified
        and checkpoint_validation.get("reason") == "cross_validation_unavailable"
        else "blocked_cross_validation_gate"
    )
    return {
        "schema_version": 1,
        "verification": "gfs_cross_action_planning_authority",
        "protocol_id": protocol.get("protocol_id"),
        "status": status,
        "code_ready": bool(contract_verified and code_paths_verified),
        "contract_verified": contract_verified,
        "code_paths_verified": code_paths_verified,
        "checkpoint_identity_verified": checkpoint_identity_verified,
        "checkpoint_cross_validation": checkpoint_validation,
        "cross_planning_authorized": authority,
        "runtime_cross_quality": (
            float(checkpoint_validation.get("quality", 0.0)) if authority else 0.0
        ),
        "claim_scope": protocol.get("claim_scope"),
        "training_executed": bool(execution.get("training_executed")),
        "matches_executed": int(execution.get("matches_executed", 0)),
        "provider_calls_made": bool(execution.get("provider_calls_made")),
    }


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
