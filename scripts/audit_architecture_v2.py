#!/usr/bin/env python3
"""Machine-check dependency, lifecycle, integrity, and packaging boundaries."""

from __future__ import annotations

import ast
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_engine.dataset_registry import write_json_atomic
from src.infrastructure import (
    file_sha256, portable_text_hash_matches, verify_artifact_manifest,
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.add(node.module)
    return values


def _forbidden_imports(area: str, forbidden: tuple[str, ...]) -> list[str]:
    violations: list[str] = []
    for path in (ROOT / "src" / area).rglob("*.py"):
        for imported in _imports(path):
            if any(
                imported == prefix or imported.startswith(prefix + ".")
                for prefix in forbidden
            ):
                violations.append(
                    f"{path.relative_to(ROOT).as_posix()} -> {imported}"
                )
    return sorted(violations)


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def _artifact_inside_root(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return None
    return resolved


def main() -> int:
    domain_product_violations: list[str] = []
    for area in ("match_engine", "simulation", "data_engine", "memory_engine"):
        domain_product_violations.extend(
            _forbidden_imports(area, ("src.product", "src.training"))
        )
    infrastructure_violations = _forbidden_imports(
        "infrastructure",
        tuple(f"src.{name}" for name in (
            "agents", "data_engine", "match_engine", "memory_engine",
            "product", "simulation", "training",
        )),
    )
    training_violations = _forbidden_imports(
        "training",
        ("src.product", "src.data_engine", "src.match_engine",
         "src.memory_engine", "src.simulation"),
    )
    product_training_violations = _forbidden_imports("product", ("src.training",))

    trainer = (ROOT / "scripts/train_world_model.py").read_text(encoding="utf-8")
    ensure = (ROOT / "scripts/ensure_world_model.py").read_text(encoding="utf-8")
    job = (ROOT / "src/training/job.py").read_text(encoding="utf-8")
    workspace = (ROOT / "src/product/workspace.py").read_text(encoding="utf-8")
    cli = (ROOT / "src/cli.py").read_text(encoding="utf-8")
    formal_runner = (
        ROOT / "scripts/run_formal_experiment.py"
    ).read_text(encoding="utf-8")
    formal_protocol = _read("data/evaluation/formal_experiment_protocol_v2.json")
    runtime = (ROOT / "src/simulation/runtime.py").read_text(encoding="utf-8")
    gateway = (ROOT / "src/simulation/llm_gateway.py").read_text(encoding="utf-8")
    wm_inference = (
        ROOT / "src/match_engine/world_model/inference.py"
    ).read_text(encoding="utf-8")

    registry = _read("data/training/entrypoints.json")
    discovered_trainers = {
        path.name for path in (ROOT / "scripts").glob("train_*.py")
    } | {"ensure_world_model.py"}
    registered_trainers = set((registry.get("entrypoints") or {}).keys())

    current = _read("data/releases/current.json")
    release_path = _artifact_inside_root(current.get("manifest"))
    release_pointer_ok = bool(
        release_path is not None
        and portable_text_hash_matches(
            release_path, str(current.get("manifest_sha256") or ""),
        )
    )
    release_artifacts = {"ok": False, "artifacts": 0, "failures": []}
    release_payload: dict = {}
    if release_pointer_ok and release_path is not None:
        release_payload = json.loads(release_path.read_text(encoding="utf-8-sig"))
        release_artifacts = verify_artifact_manifest(ROOT, release_payload)

    phase5 = _read("data/evaluation/phase5_research_layer_validation_v1.json")
    candidate = phase5.get("world_model_candidate") or {}
    checkpoint = _artifact_inside_root(candidate.get("checkpoint"))
    checkpoint_hash = file_sha256(checkpoint) if checkpoint and checkpoint.is_file() else None
    m1 = _read("data/evaluation/formal_ablation/M1_calibrated_decision.json")
    m1_candidate = m1.get("candidate") or {}
    candidate_identity_ok = bool(
        checkpoint_hash
        and checkpoint_hash == candidate.get("checkpoint_sha256")
        and checkpoint_hash == m1_candidate.get("checkpoint_sha256")
    )

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_find = (
        pyproject.get("tool", {}).get("setuptools", {})
        .get("packages", {}).get("find", {})
    )

    checks = {
        "domain_does_not_depend_on_product_or_training": not domain_product_violations,
        "infrastructure_has_no_upward_dependencies": not infrastructure_violations,
        "training_lifecycle_has_no_domain_or_product_dependencies": not training_violations,
        "product_does_not_import_training_implementation": not product_training_violations,
        "training_has_exclusive_process_lease": "FileLease" in job,
        "training_stop_forces_exact_checkpoint": (
            "or stop_requested" in trainer
            and "cannot stop safely before checkpointing" in job
        ),
        "training_resume_binds_data_and_code_identity": (
            "dataset_identity_sha256" in trainer
            and "training_contract_sha256" in trainer
        ),
        "training_attempts_are_distinguishable": "attempt_id" in job,
        "training_requires_explicit_authority": (
            '"--train"' in ensure and "requires explicit --train" in ensure
        ),
        "all_training_entrypoints_are_classified": (
            discovered_trainers == registered_trainers
        ),
        "studio_uses_context_local_configuration": (
            "environment_override(values)" in workspace
            and "ContextVar" in runtime
            and "os.environ.update" not in workspace
        ),
        "studio_has_transactional_run_journal": (
            "next_match_index" in workspace
            and "simulation_completed_at" in workspace
            and '"state": "failed"' in workspace
        ),
        "studio_exposes_one_guided_end_to_end_workflow": (
            "def workflow(" in workspace
            and "def cmd_studio_run(" in cli
            and all(state in workspace for state in (
                '"blocked"', '"ready_to_run"', '"running"', '"review"',
            ))
        ),
        "formal_experiment_is_preregistered_and_compute_bounded": (
            formal_protocol.get("state") == "preregistered_not_executed"
            and formal_protocol.get("design", {}).get("pairs_total") == 30
            and formal_protocol.get("design", {}).get("runs_total") == 60
            and not formal_protocol.get("integrity", {}).get("interim_analysis")
            and not formal_protocol.get("integrity", {}).get("optional_stopping")
            and formal_protocol.get("analysis", {}).get("secondary_role")
            == "descriptive_only_no_promotion_claim"
        ),
        "formal_experiment_fails_closed_on_identity_or_partial_evidence": (
            "cannot resume after protocol, checkpoint, or code drift" in formal_runner
            and "experiment must be completed before analysis" in formal_runner
            and "experimental units do not match exactly" in formal_runner
            and 'actions.add_argument("--execute"' in formal_runner
        ),
        "stable_release_pointer_identity_verified": release_pointer_ok,
        "stable_release_artifact_chain_verified": bool(release_artifacts.get("ok")),
        "research_checkpoint_identity_chain_verified": candidate_identity_ok,
        "world_model_required_mode_fails_closed": (
            "required world model failed to load" in wm_inference
            and "MATCH_WORLD_MODEL_REQUIRED" in workspace
        ),
        "shot_head_decision_binds_promoted_artifact": (
            "promotion_artifact_sha256" in (
                ROOT / "scripts/validate_frozen_shot_head.py"
            ).read_text(encoding="utf-8")
        ),
        "llm_pool_is_config_scoped_and_locked": (
            "dict[LLMGatewayConfig" in gateway and "_GATEWAY_LOCK" in gateway
        ),
        "llm_success_count_requires_accepted_content": (
            gateway.index("extract_json_object(content)")
            < gateway.index("self.call_count += 1")
        ),
        "wheel_preserves_src_console_namespace": (
            pyproject.get("project", {}).get("scripts", {}).get("gfs") == "src.cli:main"
            and package_find.get("where") == ["."]
            and {"src", "src.*"}.issubset(set(package_find.get("include") or []))
        ),
    }
    report = {
        "schema_version": 2,
        "architecture": "gfs-layered-product-v2.1",
        "checks": checks,
        "all_pass": all(checks.values()),
        "violations": {
            "domain_product_or_training_imports": domain_product_violations,
            "infrastructure_upward_imports": infrastructure_violations,
            "training_upward_imports": training_violations,
            "product_training_imports": product_training_violations,
            "unclassified_training_entrypoints": sorted(
                discovered_trainers - registered_trainers
            ),
            "stale_training_registry_entries": sorted(
                registered_trainers - discovered_trainers
            ),
        },
        "integrity": {
            "active_release": current.get("active"),
            "release_pointer_verified": release_pointer_ok,
            "release_artifacts": release_artifacts,
            "world_model_checkpoint": candidate.get("checkpoint"),
            "world_model_checkpoint_sha256": checkpoint_hash,
            "world_model_identity_chain_verified": candidate_identity_ok,
            "formal_m1_decision": m1.get("decision"),
        },
        "deployment_note": (
            "This mutable development worktree is not itself a newly promoted release; "
            "the active frozen release remains the current.json authority."
        ),
    }
    target = ROOT / "data/evaluation/architecture_v2_audit.json"
    write_json_atomic(target, report)
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
