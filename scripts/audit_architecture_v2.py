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

from src.data_engine.dataset_registry import write_json_atomic  # noqa: E402
from src.infrastructure import (  # noqa: E402
    code_identity_manifest,
    file_sha256,
    portable_text_hash_matches,
    verify_artifact_manifest,
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


_PY_RANDOM_DRAWS = {
    "betavariate", "choice", "choices", "expovariate", "gammavariate",
    "gauss", "getrandbits", "normalvariate", "randint", "random",
    "randrange", "sample", "shuffle", "triangular", "uniform",
}
_NP_RANDOM_DRAWS = {
    "beta", "choice", "gamma", "normal", "permutation", "poisson",
    "rand", "randint", "randn", "random", "shuffle", "uniform",
}


def _direct_global_rng_draws(paths: list[Path]) -> list[str]:
    """Find draws coupled to process-global RNG state in runtime modules."""
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            python_global = (
                isinstance(owner, ast.Name) and owner.id == "random"
                and node.func.attr in _PY_RANDOM_DRAWS
            )
            numpy_global = (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "np" and owner.attr == "random"
                and node.func.attr in _NP_RANDOM_DRAWS
            )
            if python_global or numpy_global:
                violations.append(
                    f"{path.relative_to(ROOT).as_posix()}:{node.lineno} -> "
                    f"{ast.unparse(node.func)}"
                )
    return sorted(violations)


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def _class_methods(source: str, class_name: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _class_bases(source: str, class_name: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {ast.unparse(base) for base in node.bases}
    return set()


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
    stochastic_runtime_paths = sorted(
        (ROOT / "src" / "simulation").rglob("*.py")
    ) + sorted((ROOT / "src" / "memory_engine").rglob("*.py")) + [
        ROOT / "src" / "app.py", ROOT / "src" / "journey_router.py",
    ]
    global_rng_draw_violations = _direct_global_rng_draws(
        stochastic_runtime_paths,
    )

    trainer = (ROOT / "scripts/train_world_model.py").read_text(encoding="utf-8")
    ensure = (ROOT / "scripts/ensure_world_model.py").read_text(encoding="utf-8")
    job = (ROOT / "src/training/job.py").read_text(encoding="utf-8")
    workspace = (ROOT / "src/product/workspace.py").read_text(encoding="utf-8")
    workspace = " ".join(workspace.split())
    workspace_session = (
        ROOT / "src/product/workspace_session.py"
    ).read_text(encoding="utf-8")
    workspace_session_tests = (
        ROOT / "tests/test_product_workspace_session.py"
    ).read_text(encoding="utf-8")
    workspace_evidence = (
        ROOT / "src/product/workspace_evidence.py"
    ).read_text(encoding="utf-8")
    workspace_evidence_tests = (
        ROOT / "tests/test_product_workspace_evidence.py"
    ).read_text(encoding="utf-8")
    web = (ROOT / "src/product/web.py").read_text(encoding="utf-8")
    decision_ledger = (
        ROOT / "src/product/decision_ledger.py"
    ).read_text(encoding="utf-8")
    world_state_evidence = (
        ROOT / "src/product/world_state_evidence.py"
    ).read_text(encoding="utf-8")
    decision_advice = (
        ROOT / "src/product/decision_advice.py"
    ).read_text(encoding="utf-8")
    manager_future_review = (
        ROOT / "src/product/manager_future_review.py"
    ).read_text(encoding="utf-8")
    world_model_fork_set = (
        ROOT / "src/product/world_model_fork_set.py"
    ).read_text(encoding="utf-8")
    product_comparison = (
        ROOT / "src/product/comparison.py"
    ).read_text(encoding="utf-8")
    product_tasks = (
        ROOT / "src/product/tasks.py"
    ).read_text(encoding="utf-8")
    product_recovery = (
        ROOT / "src/product/recovery.py"
    ).read_text(encoding="utf-8")
    production_validation = (
        ROOT / "scripts/run_production_validation.py"
    ).read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "deploy/compose.yaml").read_text(encoding="utf-8")
    recovery_verification = _read(
        "data/evaluation/product_recovery_verification_v1.json"
    )
    recovery_identity_files = (
        "src/infrastructure/atomic_io.py",
        "src/product/workspace.py",
        "src/product/workspace_session.py",
        "src/product/workspace_evidence.py",
        "src/product/manager_world_story.py",
        "src/product/recovery.py",
        "src/product/web.py",
        "src/cli.py",
        "scripts/verify_product_recovery.py",
    )
    manager_advisor_study = (
        ROOT / "scripts/manager_advisor_study.py"
    ).read_text(encoding="utf-8")
    manager_advisor_protocol = _read(
        "data/evaluation/manager_advisor_protocol_v1.json"
    )
    action_adoption_study = (
        ROOT / "scripts/action_adoption_study.py"
    ).read_text(encoding="utf-8")
    action_adoption_controller = (
        ROOT / "src/match_engine/world_model/action_adoption.py"
    ).read_text(encoding="utf-8")
    world_model_planner = (
        ROOT / "src/match_engine/world_model/planner.py"
    ).read_text(encoding="utf-8")
    cross_validation = (
        ROOT / "src/match_engine/world_model/cross_validation.py"
    ).read_text(encoding="utf-8")
    ball_path_logger = (
        ROOT / "src/match_engine/ball_path_logger.py"
    ).read_text(encoding="utf-8")
    action_codec = (
        ROOT / "src/match_engine/world_model/action_codec.py"
    ).read_text(encoding="utf-8")
    product_replay = (
        ROOT / "src/product/replay.py"
    ).read_text(encoding="utf-8")
    cross_protocol = _read(
        "data/evaluation/cross_action_validation_protocol_v1.json"
    )
    cross_verification = _read(
        "data/evaluation/cross_action_validation_verification_v1.json"
    )
    action_engine = (
        ROOT / "src/match_engine/action_engine.py"
    ).read_text(encoding="utf-8")
    action_adoption_protocol = _read(
        "data/evaluation/action_adoption_protocol_v1.json"
    )
    manager_intelligence = (
        ROOT / "src/product/manager_intelligence.py"
    ).read_text(encoding="utf-8")
    official_action_execution = (
        ROOT / "src/product/world_model_action_execution.py"
    ).read_text(encoding="utf-8")
    product_reporting = (
        ROOT / "src/product/reporting.py"
    ).read_text(encoding="utf-8")
    manager_world_thread = (
        ROOT / "src/product/manager_world_thread.py"
    ).read_text(encoding="utf-8")
    manager_intervention_workspace = (
        ROOT / "src/product/manager_intervention_workspace.py"
    ).read_text(encoding="utf-8")
    manager_world_navigator = (
        ROOT / "src/product/manager_world_navigator.py"
    ).read_text(encoding="utf-8")
    manager_world_story = (
        ROOT / "src/product/manager_world_story.py"
    ).read_text(encoding="utf-8")
    match_micro_runner = (
        ROOT / "src/match_engine/match_micro_runner.py"
    ).read_text(encoding="utf-8")
    match_affective_runner = (
        ROOT / "src/match_engine/match_affective_runner.py"
    ).read_text(encoding="utf-8")
    internal_signals = (
        ROOT / "src/match_engine/internal_signals.py"
    ).read_text(encoding="utf-8")
    match_pipeline = (
        ROOT / "src/simulation/match_pipeline.py"
    ).read_text(encoding="utf-8")
    tournament_finalize = (
        ROOT / "src/simulation/tournament_finalize.py"
    ).read_text(encoding="utf-8")
    cross_match_state = (
        ROOT / "src/simulation/cross_match_state.py"
    ).read_text(encoding="utf-8")
    society_continuity = (
        ROOT / "src/simulation/society_continuity.py"
    ).read_text(encoding="utf-8")
    meta_learning = (
        ROOT / "src/simulation/meta_learning.py"
    ).read_text(encoding="utf-8")
    cognitive_executor = (
        ROOT / "src/match_engine/cognitive/executor.py"
    ).read_text(encoding="utf-8")
    society_continuity_tests = (
        ROOT / "tests/test_society_continuity.py"
    ).read_text(encoding="utf-8")
    meta_learning_tests = (
        ROOT / "tests/test_meta_learning_continuity.py"
    ).read_text(encoding="utf-8")
    world_state_evidence_tests = (
        ROOT / "tests/test_product_world_state_evidence.py"
    ).read_text(encoding="utf-8")
    manager_world_thread_tests = (
        ROOT / "tests/test_product_manager_world_thread.py"
    ).read_text(encoding="utf-8")
    manager_world_story_tests = (
        ROOT / "tests/test_product_manager_world_story.py"
    ).read_text(encoding="utf-8")
    web_tests = (
        ROOT / "tests/test_product_web.py"
    ).read_text(encoding="utf-8")
    tournament_scoring = (
        ROOT / "src/simulation/tournament_scoring.py"
    ).read_text(encoding="utf-8")
    tournament_reporting = (
        ROOT / "src/simulation/tournament_reporting.py"
    ).read_text(encoding="utf-8")
    score_path = (
        ROOT / "src/simulation/score_path.py"
    ).read_text(encoding="utf-8")
    code_identity = (
        ROOT / "src/infrastructure/code_identity.py"
    ).read_text(encoding="utf-8")
    mirrored_policy_evaluation = (
        ROOT / "src/match_engine/world_model/mirrored_policy_evaluation.py"
    ).read_text(encoding="utf-8")
    m2_study = (
        ROOT / "scripts/run_m2_mirrored_policy_study.py"
    ).read_text(encoding="utf-8")
    m2_trainer = (
        ROOT / "scripts/train_world_model.py"
    ).read_text(encoding="utf-8")
    m2_validator = (
        ROOT / "scripts/validate_world_model.py"
    ).read_text(encoding="utf-8")
    m2_preflight_source = (
        ROOT / "scripts/preflight_m2_training.py"
    ).read_text(encoding="utf-8")
    m2_runtime = (
        ROOT / "src/match_engine/world_model/inference.py"
    ).read_text(encoding="utf-8")
    m2_policy_utility = (
        ROOT / "src/match_engine/world_model/policy_utility.py"
    ).read_text(encoding="utf-8")
    m2_protocol_path = ROOT / "data/evaluation/m2_mirrored_policy_protocol_v1.json"
    m2_protocol = _read("data/evaluation/m2_mirrored_policy_protocol_v1.json")
    m2_preflight = _read("data/evaluation/m2_training_preflight_v1.json")
    m2_integrity = m2_protocol.get("integrity") or {}
    m2_manifest_path = ROOT / str(
        (m2_protocol.get("candidate") or {}).get("dataset_manifest") or ""
    )
    m2_preflight_code = code_identity_manifest(
        ROOT,
        [
            *(m2_integrity.get("code_identity_files") or []),
            "src/match_engine/world_model/mirrored_policy_evaluation.py",
        ],
        mode=str(m2_integrity.get("code_identity_mode") or ""),
    )
    m2_expected_preflight_identity = {
        "protocol_path": m2_protocol_path.relative_to(ROOT).as_posix(),
        "protocol_sha256": file_sha256(m2_protocol_path),
        "manifest_path": m2_manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": file_sha256(m2_manifest_path),
        "code_sha256": m2_preflight_code,
    }
    wm_decision_support = (
        ROOT / "src/match_engine/world_model/decision_support.py"
    ).read_text(encoding="utf-8")
    agent = (ROOT / "src/simulation/agent.py").read_text(encoding="utf-8")
    agent_psychology = (
        ROOT / "src/simulation/agent_psychology.py"
    ).read_text(encoding="utf-8")
    agent_governance = (
        ROOT / "src/simulation/agent_governance.py"
    ).read_text(encoding="utf-8")
    agent_tactics = (
        ROOT / "src/simulation/agent_tactics.py"
    ).read_text(encoding="utf-8")
    agent_reflection = (
        ROOT / "src/simulation/agent_reflection.py"
    ).read_text(encoding="utf-8")
    agent_condition = (
        ROOT / "src/simulation/agent_condition.py"
    ).read_text(encoding="utf-8")
    agent_facade_methods = _class_methods(agent, "SocietyAgent")
    agent_facade_bases = _class_bases(agent, "SocietyAgent")
    agent_behavior_layers = {
        "AgentGovernanceMixin": (
            agent_governance,
            {
                "simulate_internal_game", "apply_referee_dynamics",
                "relax_referee_grievance_post_match", "update_governance_post_match",
                "_bounded_sigmoid",
            },
        ),
        "AgentTacticsMixin": (
            agent_tactics,
            {
                "_clip01", "set_tactical_controls", "refresh_tactical_vector",
                "tactical_effects", "coach_intervention",
            },
        ),
        "AgentReflectionMixin": (
            agent_reflection,
            {
                "_reflection_context_payload", "request_reflection_payload",
                "apply_reflection_payload", "perform_reflection",
                "ingest_micro_cognitive_memory", "apply_llm_reflection",
                "get_context_for_llm",
            },
        ),
        "AgentConditionMixin": (
            agent_condition,
            {
                "locker_room_tension", "rare_locker_room_explosion",
                "apply_match_wear", "recover", "get_effective_status",
            },
        ),
    }
    world_runner = (
        ROOT / "src/simulation/world_cup_runner.py"
    ).read_text(encoding="utf-8")
    public_app = (ROOT / "src/app.py").read_text(encoding="utf-8")
    tournament_checkpoint = (
        ROOT / "src/simulation/tournament_checkpoint.py"
    ).read_text(encoding="utf-8")
    tournament_match = (
        ROOT / "src/simulation/tournament_match.py"
    ).read_text(encoding="utf-8")
    tournament_lifecycle = (
        ROOT / "src/simulation/tournament_lifecycle.py"
    ).read_text(encoding="utf-8")
    tournament_state = (
        ROOT / "src/simulation/tournament_state.py"
    ).read_text(encoding="utf-8")
    tournament_transaction = (
        ROOT / "src/simulation/tournament_transaction.py"
    ).read_text(encoding="utf-8")
    tournament_runtime = (
        ROOT / "src/simulation/tournament_2026.py"
    ).read_text(encoding="utf-8")
    tournament_facade_methods = _class_methods(
        tournament_runtime, "TournamentManager",
    )
    tournament_facade_bases = _class_bases(
        tournament_runtime, "TournamentManager",
    )
    tournament_behavior_layers = {
        "TournamentLifecycleMixin": (
            tournament_lifecycle,
            {
                "_reflection_with_retry", "run_full_tournament",
                "simulate_group_stage", "update_standings",
                "resolve_advancements", "simulate_knockout_round",
                "_record_final_result",
            },
        ),
        "TournamentStateMixin": (
            tournament_state,
            {
                "_save_checkpoint", "_restore_from_checkpoint",
                "_require_run_identity",
            },
        ),
    }
    tournament_world_state = (
        ROOT / "src/simulation/world_state.py"
    ).read_text(encoding="utf-8")
    roster_loader = (
        ROOT / "src/data_engine/roster_loader.py"
    ).read_text(encoding="utf-8")
    squad_factory = (
        ROOT / "src/match_engine/squad_factory.py"
    ).read_text(encoding="utf-8")
    macro_bridge = (
        ROOT / "src/match_engine/macro_bridge.py"
    ).read_text(encoding="utf-8")
    season = (ROOT / "src/product/season.py").read_text(encoding="utf-8")
    cli = (ROOT / "src/cli.py").read_text(encoding="utf-8")
    formal_runner = (
        ROOT / "scripts/run_formal_experiment.py"
    ).read_text(encoding="utf-8")
    formal_protocol = _read("data/evaluation/formal_experiment_protocol_v2.json")
    security_protocol = _read(
        "data/evaluation/security_closure_protocol_v2.json"
    )
    security_verifier = (
        ROOT / "scripts/verify_security_closure.py"
    ).read_text(encoding="utf-8")
    completion_plan = (
        ROOT / "src/product/completion_plan.py"
    ).read_text(encoding="utf-8")
    control_plane = (
        ROOT / "src/product/control_plane.py"
    ).read_text(encoding="utf-8")
    evidence_kit = (
        ROOT / "scripts/build_excellence_evidence_kit.py"
    ).read_text(encoding="utf-8")
    human_study = (
        ROOT / "src/product/human_study.py"
    ).read_text(encoding="utf-8")
    product_validation_study = (
        ROOT / "scripts/product_validation_study.py"
    ).read_text(encoding="utf-8")
    product_value_study = (
        ROOT / "scripts/product_value_study.py"
    ).read_text(encoding="utf-8")
    study_delivery = (
        ROOT / "src/product/study_delivery.py"
    ).read_text(encoding="utf-8")
    study_session = (
        ROOT / "src/product/study_session.py"
    ).read_text(encoding="utf-8")
    product_validation_protocol = _read(
        "data/evaluation/product_validation_protocol_v1.json"
    )
    product_value_protocol = _read(
        "data/evaluation/product_value_validation_protocol_v1.json"
    )
    product_value_cases = _read(
        "data/evaluation/product_value_case_packs_v1.json"
    )
    product_value_scoring_seal = _read(
        "data/evaluation/product_value_scoring_seal_v1.json"
    )
    runtime = (ROOT / "src/simulation/runtime.py").read_text(encoding="utf-8")
    gateway = (ROOT / "src/simulation/llm_gateway.py").read_text(encoding="utf-8")
    wm_inference = (
        ROOT / "src/match_engine/world_model/inference.py"
    ).read_text(encoding="utf-8")
    manager_advice_request = workspace[
        workspace.index("def request_manager_decision_advice("):
        workspace.index("def _generate_manager_decision_advice_packet(")
    ]
    manager_decision_preview = workspace[
        workspace.index("def preview_manager_decision("):
        workspace.index("def manager_future_set_context(")
    ]

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
        "product_workspace_session_repository_is_dedicated_and_replay_complete": (
            all(token in workspace_session for token in (
                "def load_workspace_session(",
                "session_path.read_text",
                "season_history_view(session)",
                "sporting_brief_for_transition(",
                "validate_development_registry(",
                "validate_lifecycle_registry(",
                "validate_market_registry(",
                "validate_scouting_registry(",
                "validate_scouting_outcome_registry(",
                "validate_sporting_review_registry(",
                "validate_squad_registry(",
                "validate_finance_registry(",
                "validate_league_ecosystem(",
            ))
            and "src.product.workspace" not in workspace_session
            and len(workspace_session.splitlines()) <= 1100
            and all(token in workspace for token in (
                "from src.product.workspace_session import load_workspace_session",
                "def _session(self) -> dict[str, Any]: return load_workspace_session(",
                "season_history_view=self._season_history_view",
                "sporting_brief_for_transition=self._sporting_brief_for_transition",
            ))
            and "session = json.loads(self.session_path" not in workspace
            and all(token in workspace_session_tests for token in (
                "test_workspace_session_repository_loads_a_minimal_valid_session",
                "test_workspace_session_repository_fails_closed_on_registry_tampering",
                "test_product_workspace_session_facade_forwards_exact_dependencies",
                '"src/product/workspace_session.py" in CODE_IDENTITY_FILES',
            ))
        ),
        "product_workspace_evidence_projection_is_dedicated_and_read_only": (
            all(token in workspace_evidence for token in (
                "def _formal_evidence_identity(",
                "def build_workspace_evidence(",
                "def build_workspace_readiness(",
                "ArtifactResolver = Callable[[Path, Any], Path | None]",
                "validate_m2_preflight_receipt(",
                "validate_m2_candidate_receipt(",
                "portable_text_hash_matches(",
                "verify_artifact_manifest(",
                "environment_snapshot()",
            ))
            and "src.product.workspace" not in workspace_evidence
            and len(workspace_evidence.splitlines()) <= 900
            and all(token in workspace for token in (
                "from src.product.workspace_evidence import ( build_workspace_evidence, build_workspace_readiness, )",
                "def evidence(self) -> dict[str, Any]: return build_workspace_evidence(",
                "def readiness(self) -> dict[str, Any]: return build_workspace_readiness(",
                "evidence=self.evidence()",
                "from src.product.workspace_evidence import ( # noqa: F401 _formal_evidence_identity, )",
            ))
            and "def read(relative: str) -> dict:" not in workspace
            and "LLMGatewayConfig" not in workspace
            and all(token in workspace_evidence_tests for token in (
                "test_workspace_evidence_repository_fails_closed_without_authority",
                "test_product_workspace_evidence_facade_forwards_exact_dependencies",
                "test_product_workspace_readiness_facade_uses_current_evidence_once",
                '"src/product/workspace_evidence.py" in CODE_IDENTITY_FILES',
            ))
        ),
        "society_agent_psychology_is_dedicated_and_inherited": (
            "from src.simulation.agent_psychology import AgentPsychologyMixin"
            in agent
            and "    AgentPsychologyMixin," in agent
            and all(token in agent_psychology for token in (
                "class AgentPsychologyMixin:",
                "def _initialize_psychology_from_history(",
                "def _initialize_latent_states(",
                "def _project_latents_to_states(",
                "def _appraise_event(",
                "def _emotion_from_appraisal(",
                "def _coping_from_appraisal_emotion(",
                "def _memory_salience(",
                "def hidden_state(",
            ))
            and all(token not in agent for token in (
                "def _initialize_psychology_from_history(",
                "def _initialize_latent_states(",
                "def _project_latents_to_states(",
                "def _appraise_event(",
                "def _emotion_from_appraisal(",
                "def _coping_from_appraisal_emotion(",
                "def _memory_salience(",
                "def hidden_state(",
            ))
        ),
        "society_agent_facade_has_dedicated_behavior_layers": (
            set(agent_behavior_layers).issubset(agent_facade_bases)
            and all(
                methods.issubset(_class_methods(source, mixin))
                for mixin, (source, methods) in agent_behavior_layers.items()
            )
            and all(
                not methods.intersection(agent_facade_methods)
                for _, methods in agent_behavior_layers.values()
            )
            and agent_facade_methods == {
                "_finite", "__init__", "_infer_region", "_infer_style_archetype",
            }
        ),
        "tournament_manager_facade_has_dedicated_lifecycle_and_state": (
            set(tournament_behavior_layers).issubset(tournament_facade_bases)
            and all(
                methods.issubset(_class_methods(source, mixin))
                for mixin, (source, methods) in tournament_behavior_layers.items()
            )
            and all(
                not methods.intersection(tournament_facade_methods)
                for _, methods in tournament_behavior_layers.values()
            )
            and tournament_facade_methods == {
                "__init__", "_match_key", "_map_micro_score",
                "_stage_pressure", "_normalize_weights",
                "_stage_referee_distribution", "_sample_referee_profile",
            }
        ),
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
        "web_request_gateway_is_layered_and_security_ordered": (
            all(token in web for token in (
                "def _health_response(",
                "def _require_allowed_access(",
                "def _authentication_response(",
                "def _get_response(",
                "def _post_response(",
                "def _is_method_restricted_path(",
                "payload_handler(self._read_json(environ))",
                "queued_handler(environ, self._read_json(environ))",
            ))
            and web.index("health = self._health_response(")
            < web.index("self._require_allowed_access(environ)")
            < web.index("authentication = self._authentication_response(")
            and web.index(
                "self._require_csrf(environ)", web.index("def _post_response(")
            ) < web.index("payload_handler(self._read_json(environ))")
        ),
        "studio_status_projection_is_typed_layered_and_noncausal": (
            all(token in web for token in (
                "def _unconfigured_studio_status(",
                "def _attach_status_artifact_urls(",
                "def _library_matches(",
                "def _fork_set_library_item(",
                "def _fork_propagation_for_web(",
                "def _fork_library_item(",
                "def _pair_library_item(",
                "def _study_library_item(",
                "def _library_task_groups(",
                "def _attach_manager_world_views(",
                '"ranking_performed": False',
                '"downstream_causal_attribution_authorized": False',
                '"match_outcome_causality": False',
                '"real_football_causality": False',
                "self._attach_manager_world_views(season, library_fork_sets)",
            ))
        ),
        "studio_restore_is_crash_convergent": (
            all(token in product_recovery for token in (
                'RESTORE_TRANSACTION_RELATIVE =',
                "def _recover_interrupted_restore_unlocked(",
                '"old_sha256":',
                "_durable_replace(preparing, self.restore_transaction_root)",
                "force_rollback=True",
                '"completed_committed_restore"',
                '"rolled_back_interrupted_restore"',
                "_remove_restore_transaction_unlocked",
            ))
            and web.index(
                "ProductRecovery(resolved_root).recover_interrupted_restore()"
            ) < web.index("queue.recover_running()")
            and 'def cmd_studio_recover(' in cli
            and recovery_verification.get("passed") is True
            and recovery_verification.get("matches_executed") == 0
            and recovery_verification.get("training_executed") is False
            and recovery_verification.get("code_sha256") == {
                relative: file_sha256(ROOT / relative)
                for relative in recovery_identity_files
            }
            and all(
                recovery_verification.get("checks", {}).get(name) is True
                for name in (
                    "real_process_crash_was_injected",
                    "mid_restore_transaction_was_durable",
                    "mid_restore_crash_rolled_back_exactly",
                    "rollback_recovery_is_idempotent",
                    "post_switch_crash_was_injected",
                    "fully_switched_restore_was_completed",
                    "atomic_replace_failure_preserved_document",
                )
            )
        ),
        "studio_exposes_one_guided_end_to_end_workflow": (
            "def workflow(" in workspace
            and "def cmd_studio_run(" in cli
            and all(state in workspace for state in (
                '"blocked"', '"ready_to_start_season"', '"running"',
                '"season_setup"', '"season_decision"',
                '"season_matchday_ready"', '"season_planning"', '"review"',
            ))
            and all(action in workspace for action in (
                '"start_season"', '"freeze_player_promises"',
                '"submit_manager_decision"', '"advance_season_matchday"',
                '"review_sporting_plan"',
            ))
        ),
        "studio_progressively_discloses_product_areas": (
            all(token in web for token in (
                'data-workspace-view="career"',
                'data-workspace-view="lab"',
                'data-workspace-view="evidence"',
                'data-workspace-view="operations"',
                "function applyWorkspaceArea(",
                "function workflowWorkspaceArea(",
                "function renderWorkspaceAreas(",
                "workspaceAreaExplicit",
                "workflowAction._area=workflowAreaByAction[action]",
            ))
            and ".workspace-area-hidden { display:none!important }" in web
            and "innerHTML" not in web
        ),
        "studio_manager_journey_is_state_derived_and_unified": (
            all(token in web for token in (
                'id="manager-product-journey"',
                "function renderManagerProductJourney(",
                "season?.matchday_command_center",
                "renderUnifiedWorkflowWithoutManagerJourney",
                "item.dataset.stage=id",
                "item.dataset.status=status",
                "from src.product.manager_world_story import (",
                "build_manager_world_story(world_navigator)",
                "validate_manager_world_story(world_story, navigator=world_navigator)",
                'season["manager_world_story"] = world_story',
                'command["manager_world_story"] = world_story',
                'id="manager-world-story"',
                "function renderManagerWorldStory(",
                "function renderManagerWorldNavigatorBase(season)",
                "const MANAGER_WORLD_RENDER_STAGES=Object.freeze([",
                "function renderManagerWorldNavigator(season){for(const renderStage of ",
            ))
            and "def _manager_world_story_for_web(" not in web
            and "renderManagerWorldNavigatorWithout" not in web
            and "renderManagerWorldReviewedFutureContinuityWithout" not in web
            and web.count("function renderManagerWorldNavigator(season)") == 1
            and "from src.product.web" not in manager_world_story
            and manager_world_story.count("\n") < 260
            and all(token in manager_world_story for token in (
                "def build_manager_world_story(",
                "def validate_manager_world_story(",
                'view_mode = "active_chapter"',
                '"previous_completed": previous_completed',
                'current.get("current_chapter_identity")',
                'current.get("workflow_state") != "season_complete"',
                '"same_chapter_evidence": True',
                '"outcome_improvement_authorized": False',
                '"causal_effect_authorized": False',
                "if dict(story) != expected:",
            ))
            and all(token in manager_world_story_tests for token in (
                "test_world_story_uses_one_latest_chapter_and_stays_noncausal",
                "test_world_story_exposes_current_review_without_fake_result",
                "test_world_story_prioritizes_active_chapter_and_binds_previous_world",
                "test_world_story_empty_completed_season_has_no_invented_chapter",
                "test_world_story_unavailable_state_is_fresh_and_fail_closed",
                "test_world_story_validation_rejects_projection_drift",
                '"src/product/manager_world_story.py" in CODE_IDENTITY_FILES',
            ))
            and all(token in web_tests for token in (
                'reviewed_projection["manager_world_story"]',
                '"story.view_mode===\'active_chapter\'"',
                '"const openSource=source.navigable?source:previous?.source"',
            ))
            and all(token in web for token in (
                "story.view_mode==='active_chapter'",
                "const openSource=source.navigable?source:previous?.source",
                "openSource?.navigable&&openSource.chapter_identity",
            ))
            and "innerHTML" not in web
        ),
        "action_adoption_smoke_is_four_arm_and_zero_training": (
            all(token in action_adoption_study for token in (
                'SMOKE_ARM_IDS = (',
                '"M0_no_advisor"',
                '"C1_rule_fallback"',
                '"M1_predict_only"',
                '"M1_action_policy"',
                "def smoke(",
                '"formal_progress_modified": False',
                '"training_executed": False',
                '"provider_calls_made": False',
                "def export_analysis_artifacts(",
                'actions.add_argument("--smoke"',
            ))
            and action_adoption_protocol.get("state")
            == "preregistered_not_executed"
            and set(action_adoption_protocol.get("outputs") or {}) == {
                "progress", "decision", "rows_csv", "summary_markdown", "smoke",
            }
            and action_adoption_protocol.get("current_execution", {}).get(
                "runs_executed"
            ) == 0
        ),
        "action_adoption_progress_is_fixed_schedule_and_fail_closed": (
            all(token in action_adoption_study for token in (
                "def validate_progress(",
                '"rows_follow_exact_schedule_prefix"',
                '"analysis_metrics_are_finite"',
                '"action_counts_are_bounded"',
                '"arm_order_is_resumable"',
                '"required_complete_budget"',
                '"blocked_invalid_progress"',
                '"inspect_invalid_progress"',
                "candidate_rows = [*rows, dict(row)]",
                "_require_valid_progress(validate_progress(",
            ))
            and action_adoption_study.count(
                "_require_valid_progress(validate_progress("
            ) >= 3
        ),
        "world_model_action_authority_uses_validated_feasible_simplex": (
            all(token in action_adoption_controller for token in (
                "validated_action_simplex_v3",
                "applied_policy_actions",
                "action_signal_breakdown",
                "applied_action_authority",
                "model_target_action_probability",
                "str(action) not in feasible",
                "blend_weight = max(item[3] for item in signals)",
            ))
            and all(token in world_model_planner for token in (
                "validated_shot_vs_continuation_advantage",
                '"model_advantage": effective_advantage',
                'gates["shot"]["reason"] = "action_infeasible"',
                '"no_action_specific_validation"',
            ))
            and all(token in action_engine for token in (
                '_ACTION_LABELS = ("pass", "shot", "cross", "hold")',
                "def _sample_action(",
                "def _record_policy_result(",
                "def _execute_sampled_action(",
                "mask_infeasible_action_probabilities(",
                "mix_direct_action_probabilities(",
                "sampling_uniform = float(rng.random())",
                "counterfactual_baseline_action = sample_action_from_uniform(",
            ))
            and all(token in workspace_evidence for token in (
                "def _formal_evidence_identity(",
                '"result_identity_verified": mechanism_current',
                '"result_identity_verified": outcome_current',
                '"stale_current_code_identity"',
                '"result_applicable_to_current_code"',
            ))
            and all(token in web for token in (
                "function renderActionAdoptionCurrentCodeEvidence(studio)",
                "mechanism.result_identity_verified",
                "outcome.result_identity_verified",
            ))
        ),
        "action_adoption_product_render_pipeline_is_explicit": (
            all(token in web for token in (
                "function renderActionAdoptionBase(studio)",
                "const ACTION_ADOPTION_RENDER_STAGES=Object.freeze([",
                "renderActionAdoptionActionSignals",
                "renderActionAdoptionShotValidation",
                "renderActionAdoptionHoldReference",
                "renderActionAdoptionFormalEvidence",
                "renderActionAdoptionCurrentCodeEvidence",
                "renderActionAdoptionManagerProtocol",
                "renderActionAdoptionM2Outcome",
                "renderActionAdoptionM2ResearchControl",
                "function renderActionAdoption(studio){for(const renderStage of ",
            ))
            and "renderActionAdoptionWithout" not in web
            and web.count("function renderActionAdoption(studio)") == 1
        ),
        "hold_is_reference_only_not_direct_model_authority": (
            all(token in action_adoption_controller for token in (
                '_DIRECTLY_VALIDATED_ACTIONS = frozenset({"pass", "shot", "cross"})',
                '_REFERENCE_ACTION = "hold"',
                'or str(action) not in _DIRECTLY_VALIDATED_ACTIONS',
                '"reference_action_directly_authorized": False',
                '"reference_action_breakdown": {',
                '"direct_signal_opportunities": 0',
                'recommended != "none" and actual == recommended',
                "influenced = bool(policy_signals)",
                '"mean_primary_signal_probability_shift"',
            ))
            and all(token in world_model_planner for token in (
                '"reference" if action == "hold"',
                '"counterfactual_reference_only" if action == "hold"',
                '"direct_action_authorized": False',
            ))
            and all(token in workspace for token in (
                'value.get("reference_action_breakdown")',
                '"redistribution_opportunities"',
                '"mean_probability_gain"',
            ))
            and all(token in web for token in (
                "function renderActionAdoptionHoldReference(studio)",
                "reference.redistribution_opportunities",
                "reference.mean_probability_gain",
            ))
        ),
        "studio_previews_authoritative_decision_effects_without_persistence": (
            workspace.count("self._prepare_manager_decision(") >= 2
            and "def preview_manager_decision(" in workspace
            and "season = copy.deepcopy(source)" in workspace
            and "_atomic_json(self.session_path, session)"
            not in manager_decision_preview
            and all(token in web for token in (
                '"/api/v1/seasons/decision-preview": self._preview_season_decision',
                "function managerDecisionPayload(includeRevision=false)",
                "function refreshManagerDecisionPreview()",
                "++managerPreviewSequence",
                "payload.expected_revision=managerPreviewBaseRevision",
                "不预测比分、胜率",
            ))
        ),
        "studio_replays_manager_decision_lifecycle_without_second_state": (
            "def build_manager_decision_ledger(" in decision_ledger
            and "manager_season_profile(state)" in decision_ledger
            and "commitment_progress_from_evidence(" in decision_ledger
            and "player_promise_progress(" in decision_ledger
            and "execution_identity_or_boundary_mismatch" in decision_ledger
            and "payload[\"entry_identity\"] = _identity(payload)" in decision_ledger
            and "ledger[\"ledger_identity\"] = _identity(ledger)" in decision_ledger
            and "per_fixture_persisted_state_snapshot_not_retained" in decision_ledger
            and "build_manager_decision_ledger(" in workspace
            and "renderManagerDecisionLedgerWithout" not in web
            and web.count("function renderManagerDecisionLedger(season)") == 1
            and "live_window: int = 8" in workspace
            and all(token in web for token in (
                'id="manager-decision-ledger"',
                "function renderManagerDecisionLedgerBase(season)",
                "const MANAGER_DECISION_LEDGER_RENDER_STAGES=Object.freeze([",
                "function renderManagerDecisionLedger(season)",
                "单场赛果不证明决策效果",
            ))
        ),
        "managed_fixtures_bind_three_phase_world_state_evidence": (
            all(token in world_state_evidence for token in (
                "def capture_world_state(",
                "def build_fixture_world_state_transition(",
                "def validate_fixture_world_state_transition(",
                '"before_match":', '"after_match":', '"after_recovery":',
                "expected = _build_fixture_world_state_transition(",
                "len(players) > 128",
                "deterministic_team_baseline",
                "deterministic_roster_baseline",
                "persisted_carryover",
            ))
            and all(token in workspace for token in (
                "decision is not None",
                '"world_state_before"',
                '"world_state_transition"',
                "after_recovery = capture_world_state(",
                "validate_fixture_world_state_transition(",
            ))
            and all(token in season for token in (
                "validate_team_state_snapshot(before_state[team], team=team)",
                "validate_fixture_world_state_transition(",
                "fixture recovery-state evidence is inconsistent",
                '"world_state_transition": copy.deepcopy(',
            ))
            and "persistent_team_state_delta" in decision_ledger
            and all(token in web for token in (
                "世界状态：比赛后疲劳",
                "比赛日恢复尚未结算",
                "伤停为模拟状态",
            ))
        ),
        "manager_action_adoption_uses_identity_bound_world_model_advice": (
            all(token in decision_advice for token in (
                "def build_manager_decision_advice(",
                "def validate_manager_decision_advice(",
                "def build_manager_advice_adoption(",
                "every playable tactic once",
                'runtime_signature != f"sha256:{checkpoint_sha256}"',
                "recommendation replay mismatch",
                "adopted world-model recommendation does not match tactic",
            ))
            and all(token in wm_decision_support for token in (
                '"team_identity" if name == "team_identity"',
                "native_tactical_vector",
                "_tactical_action_weights_from_vector",
            ))
            and "self.random_root_seed = int(random_root_seed)" in agent
            and "self._initialization_rng = initialization_rng or named_py_rng(" in agent
            and "root_seed=root_seed" in world_runner
            and all(token in workspace for token in (
                "def request_manager_decision_advice(",
                "def _generate_manager_decision_advice_packet(",
                'if self.config.mode == "stable":',
                'target["manager_decision_advice"] = advice',
                'target["manager_advice_adoption"] = build_manager_advice_adoption(',
                "manager advice is stale; request fresh advice",
                "candidate_presets=tuple(PLAYABLE_TACTICS)",
            ))
            and all(token in season for token in (
                "validate_manager_decision_advice(advice)",
                "validate_manager_advice_adoption(",
                'fixture.get("manager_decision_advice")',
                'fixture.get("manager_advice_adoption")',
            ))
            and "world_model_decision_support" in decision_ledger
            and all(token in web for token in (
                '"/api/v1/seasons/decision-advice": self._request_season_decision_advice',
                'id="manager-world-model-advice"',
                "function requestManagerWorldModelAdvice()",
                "function adoptCurrentManagerAdvice()",
                "payload.advice_adoption=",
                "建议、经理选择和赛果分别取证",
            ))
        ),
        "manager_advisor_evidence_is_replayable_and_preregistered": (
            all(token in decision_ledger for token in (
                "def world_model_advisor_summary(",
                "def validate_manager_decision_ledger(",
                "manager decision ledger summary replay mismatch",
                '"outcome_effect_estimate": None',
                '"causal_effect_authorized": False',
            ))
            and manager_advisor_protocol.get("state")
            == "preregistered_collection_not_started"
            and manager_advisor_protocol.get("analysis", {}).get(
                "fixed_information_windows"
            ) == [12, 24, 48]
            and manager_advisor_protocol.get("analysis", {}).get(
                "causal_effect_authorized"
            ) is False
            and manager_advisor_protocol.get("decision_rules", {}).get(
                "product_or_academic_promotion_authorized"
            ) is False
            and all(token in manager_advisor_study for token in (
                "validate_manager_decision_ledger(ledger)",
                '"state": "insufficient_evidence"',
                '"state": "window_not_closed"',
                '"instrumentation_confirmed" if passed',
                '"matches_executed_by_analyzer": 0',
                '"outcome_effect_estimate": None',
            ))
            and all(token in workspace_evidence for token in (
                '"data/evaluation/manager_advisor_protocol_v1.json"',
                '"manager_advisor_adoption": {',
                '"results_available": (',
                '"causal_effect_authorized": (',
                '"promotion_authorized": (',
            ))
            and all(token in web for token in (
                'id="manager-advisor-evidence-summary"',
                'id="manager-advisor-protocol-evidence"',
                "evidence.adopted_recommendation",
                "evidence.reviewed_then_selected",
                "studio?.evidence?.manager_advisor_adoption",
            ))
        ),
        "manager_advice_inference_uses_optimistic_short_leases": (
            manager_advice_request.count(
                "with FileLease(self.session_lease_path, timeout=30.0):"
            ) == 3
            and manager_advice_request.index(
                "packet = self._generate_manager_decision_advice_packet("
            ) < manager_advice_request.rindex(
                "with FileLease(self.session_lease_path, timeout=30.0):"
            )
            and all(token in manager_advice_request for token in (
                '"base_revision": int(season.get("revision", 0))',
                '"advice_inputs_changed_during_generation"',
                '"retryable": True',
                "current_session.get(\"mode\") == request_state[\"mode\"]",
                "current_snapshots = capture_world_state(self.root, teams)",
                'current_snapshots[team]["source_identity"]',
                "file_sha256(checkpoint_path) == checkpoint_sha256",
                "_atomic_json(self.session_path, current_session)",
            ))
        ),
        "manager_advice_request_is_semantically_idempotent": (
            all(token in manager_advice_request for token in (
                '"existing_advice": copy.deepcopy(',
                "existing_matches_inputs = bool(",
                'existing_advice.get("issued_revision")',
                'existing_sources[team]["source_identity"]',
                'current_target.get("manager_decision_advice")',
                '"reused": True',
                'current_advice == advice',
            ))
            and manager_advice_request.index("if existing_matches_inputs:")
            < manager_advice_request.index(
                "packet = self._generate_manager_decision_advice_packet("
            )
            and manager_advice_request.index('"reused": True')
            < manager_advice_request.index(
                "packet = self._generate_manager_decision_advice_packet("
            )
        ),
        "manager_advice_execution_trace_is_runtime_bound_and_non_causal": (
            all(token in match_micro_runner for token in (
                "def _initial_tactical_execution(",
                '"native_team_vector"',
                '"locked_preset"',
                '"initial_vector": _tactical_vector_snapshot(',
                '"final_vector"] = final_vector',
                'tactical_execution=_final_tactical_execution(',
            ))
            and all(token in manager_intelligence for token in (
                "def _strict_tactical_vector(",
                "def _tactical_binding(",
                'expected_tactic == "team_identity"',
                '"initial_vector_identity": _identity(initial)',
                'payload["binding_identity"] = _identity(payload)',
                '"tactical_execution_binding_mismatch"',
            ))
            and all(token in decision_ledger for token in (
                "def _advisor_execution_trace(",
                '"recommendation_executed"',
                '"reviewed_alternative_executed"',
                '"direct_recommendation_execution"',
                '"outcome_effect_estimate": None',
                '"causal_effect_authorized": False',
                '"manager advisor execution trace replay mismatch"',
            ))
            and all(token in web for token in (
                "function renderManagerDecisionLedgerExecutionTrace(season)",
                "trace.runtime_binding",
                "binding.initial_vector",
                "function renderManagerIntelligenceTacticalBinding(",
                "binding.changed_controls",
            ))
        ),
        "manager_future_review_is_identity_bound_and_non_causal": (
            all(token in manager_future_review for token in (
                "def build_manager_future_review(",
                "def validate_manager_future_review(",
                '"keep_after_review", "revise_after_review"',
                '"best_time_recommendation": False',
                '"match_outcome_causality": False',
                '"outcome_effect_estimate": None',
                '"causal_effect_authorized": False',
                "manager future review identity mismatch",
            ))
            and all(token in workspace for token in (
                "def review_manager_future_set(",
                "validate_manager_future_context(",
                "target.setdefault(",
                "manager_future_reviews",
                "_atomic_json(self.session_path, session)",
            ))
            and "from src.product.tasks" not in workspace
            and all(token in season for token in (
                'actual.get("manager_future_reviews")',
                "validate_manager_future_review(",
                '"manager_future_reviews": copy.deepcopy(',
            ))
            and all(token in decision_ledger for token in (
                "def world_model_future_review_summary(",
                '"world_model_future_reviews": future_reviews,',
                '"causal_effect_authorized": False',
            ))
            and all(token in web for token in (
                "/api/v1/seasons/world-model-future-review",
                "reviewManagerFutureEvidence",
                "keep_after_review",
                "revise_after_review",
                '"second_persisted_season_state": False',
            ))
        ),
        "manager_future_scenarios_are_replayable_and_product_visible": (
            all(token in world_model_fork_set for token in (
                "def project_fork_set_scenario_evidence(",
                "def validate_fork_set_scenario_evidence(",
                "def summarize_fork_set_scenario_evidence(",
                '"scenario_identity": _identity(payload)',
                "future-set scenario evidence semantics are invalid",
                "completed fork-set scenario aggregate mismatch",
            ))
            and all(token in product_tasks for token in (
                '"scenario_evidence": (',
                "project_fork_set_scenario_evidence(fork_set)",
            ))
            and all(token in manager_future_review for token in (
                "REVIEW_SCHEMA_VERSION = 3",
                'payload["scenario_evidence"] = scenarios',
                "manager future review scenario aggregate is inconsistent",
            ))
            and all(token in web for token in (
                '"scenario_evidence": scenario_evidence',
                "function renderManagerFutureSetsScenarioEvidence(",
                "function renderManagerDecisionLedgerFutureScenarioEvidence(season)",
                "scenario.scenario_identity",
                "时点之间不排名",
            ))
        ),
        "manager_future_action_examples_are_bounded_and_non_causal": (
            all(token in world_model_fork_set for token in (
                "MAX_SCENARIO_MECHANISM_EXAMPLES = 3",
                "MECHANISM_WINDOW_SECONDS = (30, 120)",
                "def _project_mechanism_examples(",
                "def _validate_mechanism_examples(",
                '"downstream_causal_attribution_authorized": False',
                '"causal_effect_authorized": False',
                "future-set mechanism window identity mismatch",
                "future-set mechanism example identity mismatch",
            ))
            and all(token in manager_future_review for token in (
                "REVIEW_SCHEMA_VERSION = 3",
                "manager future review scenario version mismatch",
            ))
            and all(token in decision_ledger for token in (
                '"retained_mechanism_examples": len(mechanism_examples)',
                '"locally_attributable_mechanism_examples": sum(',
                '"examples_with_descriptive_windows": sum(',
            ))
            and all(token in web for token in (
                "function appendFutureMechanismExamples(",
                "function renderManagerFutureSetsMechanismExamples(",
                "function renderManagerDecisionLedgerMechanismExamples(season)",
                "不授予下游因果",
                "v3_preserves_bounded_action_mechanism_examples",
            ))
            and "innerHTML" not in web
        ),
        "counterfactual_propagation_preserves_cross_and_v2_signal_semantics": (
            all(token in product_comparison for token in (
                'event_type not in {"pass", "shot", "cross"}',
                '"crosses": sum(event["type"] == "cross"',
                '"cross": "cross"',
                '"probability_policy_version": _safe_text(',
                '"hold_reference_redistributed": bool(',
                '"hold_reference_probability_delta": (',
            ))
            and all(token in world_model_fork_set for token in (
                "MECHANISM_WINDOW_METRICS_V1 = (",
                '"actions", "passes", "crosses", "shots"',
                "example_schema not in {1, 2}",
                '"probability_policy_version", "signal_mode"',
                '"hold_reference_redistributed"',
                '"schema_version": 2',
            ))
            and all(token in web for token in (
                "function appendFutureMechanismExamplesCrossMetrics(context)",
                "window?.delta?.crosses",
                "function appendFutureMechanismExamplesPolicySemantics(context)",
                "example.hold_reference_redistributed",
            ))
        ),
        "aggregate_views_preserve_cross_and_v2_action_semantics": (
            all(token in product_comparison for token in (
                '"crosses": sum(event["type"] == "cross"',
                '"passes", "crosses", "shots", "goals", "turnovers"',
                '"Crosses Δ"',
            ))
            and all(token in decision_ledger for token in (
                "def _mechanism_semantic_counts(",
                '"cross_action_mechanism_examples"',
                '"direct_preference_mechanism_examples"',
                '"suppression_only_mechanism_examples"',
                '"hold_reference_redistribution_examples"',
                '"nonzero_cross_descriptive_windows"',
            ))
            and all(token in manager_world_thread for token in (
                'terminal_review.get("cross_action_mechanism_examples")',
                'terminal_review.get("direct_preference_mechanism_examples")',
                'terminal_review.get("suppression_only_mechanism_examples")',
                'terminal_review.get("hold_reference_redistribution_examples")',
                'terminal_review.get("nonzero_cross_descriptive_windows")',
            ))
            and all(token in web for token in (
                "function renderManagerDecisionLedgerMechanismSemantics(season)",
                "function appendManagerWorldEvolutionThreadMechanismSemantics(",
                "evidence.cross_action_mechanism_examples",
                "stage.nonzero_cross_descriptive_windows",
            ))
        ),
        "manager_future_review_closes_to_runtime_selection_without_outcome_claim": (
            all(token in decision_ledger for token in (
                "def _future_review_execution_trace(",
                '"followed_by_later_review"',
                '"selected_for_fixture"',
                '"superseded_by_unreviewed_edit"',
                '"reviewed_selection_runtime_verified"',
                '"outcome_comparison_performed": False',
                "def world_model_future_review_execution_summary(",
                '"manager future review execution trace replay mismatch"',
            ))
            and all(token in web for token in (
                "function renderManagerDecisionLedgerFutureReviewExecution(season)",
                "row.future_review_execution_trace",
                "trace.runtime_binding.applied_tactic",
                "不把赛前模拟路径匹配到比分",
            ))
            and '"second_persisted_season_state": True' not in web
        ),
        "official_manager_matches_surface_world_model_action_execution": (
            all(token in official_action_execution for token in (
                "def project_world_model_action_execution(",
                "def validate_world_model_action_execution(",
                "MAX_EXAMPLES = 5",
                '"locally_attributable_action_changes_observed"',
                '"direct_ball_event_identity"',
                '"outcome_comparison_performed": False',
                '"causal_effect_authorized": False',
                "world-model official action evidence identity mismatch",
            ))
            and all(token in manager_intelligence for token in (
                "project_world_model_action_execution(",
                '"world_model_action_execution": world_model_action_execution',
                '"world_model_action_evidence_invalid"',
            ))
            and all(token in decision_ledger for token in (
                "validate_world_model_action_execution(action_execution)",
                "def world_model_official_action_execution_summary(",
                '"world_model_official_action_execution": (',
                '"outcome_effect_estimate": None',
            ))
            and all(token in web for token in (
                "function appendOfficialActionExecution(",
                "function renderManagerDecisionLedgerOfficialActionExecution(season)",
                "function renderManagerIntelligenceOfficialActionExecution(",
                "不比较比分、不证明战术质量",
            ))
            and "innerHTML" not in web
        ),
        "official_action_explanation_unifies_cross_signal_and_reference": (
            all(token in official_action_execution for token in (
                'actual not in {"pass", "shot", "cross"}',
                'and actual != "hold"',
                '"policy_signal": {',
                '"probability_policy_version": probability_policy_version',
                '"reference_action_effect": reference_projection',
                '"suppression_only"',
                '"legacy_unclassified"',
                "world-model official reference action projection is invalid",
            ))
            and all(token in product_reporting for token in (
                'record.get("primary_signal_action")',
                '"suppression_only": "suppression only; no direct recommendation"',
                "hold reference received indirect",
                "Mean direct-recommendation shift",
                "Mean primary-signal shift",
            ))
            and all(token in web for token in (
                "appendOfficialActionExecutionWithoutPolicySemantics",
                "example.policy_signal",
                "reference.received_redistributed_probability",
            ))
        ),
        "society_cognition_persists_into_next_match_and_product_thread": (
            all(token in society_continuity for token in (
                "SOCIETY_CONTINUITY_VERSION = 1",
                "MAX_STATE_BYTES = 262_144",
                "def capture_society_continuity(",
                "def validate_society_continuity_state(",
                "def apply_society_continuity(",
                "def build_society_decision_context(",
                "def society_public_snapshot(",
                "def society_public_transition(",
                '"memory_text_authority": "untrusted_context_only"',
                'canonical["state_identity"] = _identity(canonical)',
            ))
            and all(token in cross_match_state for token in (
                "society_state: Optional[Dict[str, Any]] = None",
                "apply_society_continuity(agent, carry.society_state)",
                "society_public_snapshot(",
            ))
            and all(token in match_pipeline for token in (
                "operation_id=transaction_id",
                "capture_society_continuity(",
                ".society_state = capture_society_continuity(",
            ))
            and (
                'transaction_id=f"{self.match_index}:{stage_name}:'
                in tournament_finalize
            )
            and all(token in cognitive_executor for token in (
                "build_society_decision_context",
                'trig.facts["society_continuity_context"]',
                'continuity.get("psychological_decision_modifiers")',
            ))
            and all(token in world_state_evidence for token in (
                "WORLD_STATE_SCHEMA_VERSION = 4",
                "society_public_snapshot(",
                "society_public_transition(",
                'result["society_transition"]',
            ))
            and all(token in manager_world_thread for token in (
                'society_transition=copy.deepcopy(society_transition)',
                '"society_continuity_transitions": sum(',
            ))
            and all(token in web for token in (
                "function appendManagerWorldEvolutionThreadSocietyContinuity(",
                "stage?.society_transition",
            ))
            and all(token in society_continuity_tests for token in (
                "test_matched_seed_targeted_narrative_changes_only_targeted_agent_state",
                "test_restored_psychology_reaches_next_coach_trigger_and_changes_fallback_plan",
                "test_society_state_rejects_identity_semantic_and_size_tampering",
                "test_match_settlement_captures_society_state_and_is_cognitively_idempotent",
                "test_cognitive_runtime_packet_is_compacted_before_cross_match_persistence",
            ))
        ),
        "meta_learning_is_delayed_identity_bound_and_product_visible": (
            all(token in meta_learning for token in (
                "META_PROPOSAL_VERSION = 2",
                "MIN_MATCHED_EVALUATION_UNITS = 8",
                "MIN_SIMULATOR_UTILITY = -5.0",
                "MAX_SIMULATOR_UTILITY = 5.0",
                "def validate_meta_proposal(",
                "def build_meta_evaluation_receipt(",
                "def validate_meta_evaluation_receipt(",
                "def _normalize_matched_rows(",
                "def _evaluation_statistics(",
                "def observe_agent_meta_proposals(",
                '"authorization_required": True',
                '"matched_seed_counterfactual_v1"',
                '"matched_rows": rows',
                '"matched_unit_normal_95_v1"',
                '"observed_without_effect_authority"',
                '"committed_by_matched_evaluation"',
                '"reflection_recorded_without_actionable_meta_adjustment"',
            ))
            and "self.commit(agent, proposal)" not in meta_learning
            and "def authorize_meta_proposal(" in agent_reflection
            and all(token in match_pipeline for token in (
                "observe_agent_meta_proposals(",
                "result_utility=float(np.clip(",
                "operation_id=source_transaction_id",
            ))
            and all(token in society_continuity for token in (
                'META_PUBLIC_STATUSES = (',
                '"meta_learning": _meta_public_summary(state)',
                '"meta_learning_before":',
                '"meta_learning_after":',
                '"meta_learning_delta":',
            ))
            and all(token in world_state_evidence for token in (
                "SOCIETY_WORLD_STATE_SCHEMA_VERSION = 2",
                "META_COUNTS_WORLD_STATE_SCHEMA_VERSION = 3",
                "WORLD_STATE_SCHEMA_VERSION = 4",
                "society projection version mismatch",
            ))
            and all(token in manager_world_thread for token in (
                '"meta_learning_pending": (',
                'int(latest_meta.get("shadow", 0))',
                'int(latest_meta.get("observed_pending_evaluation", 0))',
                '"meta_learning_authorized": int(',
            ))
            and all(token in web for token in (
                "function renderManagerDecisionLedgerMetaLearningSummary(season)",
                "summary?.meta_learning_pending",
                "meta_learning_after",
            ))
            and all(token in meta_learning_tests for token in (
                "test_reflection_stages_identity_bound_shadow_without_parameter_authority",
                "test_post_match_observation_is_noncausal_idempotent_and_cross_match",
                "test_matched_evaluation_receipt_is_required_and_idempotently_authorizes",
                "test_nonpositive_interval_rejects_without_mutation",
                "test_proposal_and_evaluation_tampering_fail_closed",
                "test_parameter_drift_blocks_authorization_without_partial_mutation",
                "test_empty_reflection_is_audited_without_creating_meta_authority",
                "test_boolean_lifecycle_values_fail_closed",
                'minimum_effect=-0.01',
                'out_of_range[0]["treated_utility"] = 5.01',
            ))
        ),
        "meta_learning_governance_is_content_free_replayable_and_visible": (
            all(token in society_continuity for token in (
                "META_PUBLIC_SCHEMA_VERSION = 2",
                "META_PUBLIC_RECORD_VERSION = 1",
                "def _meta_public_record(",
                "def _validate_meta_public_record(",
                "def _meta_public_summary(",
                "def _validate_meta_public_summary(",
                "def _meta_public_updates(",
                '"parameter_authority_granted": proposal.status == "committed"',
                '"identity_bound_matched_evaluation"',
                '"meta_learning_updates": _meta_public_updates(',
                "evidence and matched rows are withheld",
            ))
            and all(token in world_state_evidence for token in (
                "META_COUNTS_WORLD_STATE_SCHEMA_VERSION = 3",
                "WORLD_STATE_SCHEMA_VERSION = 4",
                '"governance" if isinstance(meta, Mapping) and "records" in meta',
                'META_COUNTS_WORLD_STATE_SCHEMA_VERSION: "counts"',
                'WORLD_STATE_SCHEMA_VERSION: "governance"',
                "fixture world-state phase schema version mismatch",
            ))
            and all(token in manager_world_thread for token in (
                "for thread, row in zip(threads, stages):",
                "latest_meta = max(meta_after",
            ))
            and all(token in web for token in (
                "function appendMetaLearningGovernance(",
                "society?.meta_learning_updates",
                "update.evaluation_after",
                "update.next_required_evidence",
                "evaluation.lower_bound_clears_threshold",
                "appendMetaLearningGovernance(details,society)",
            ))
            and all(token in world_state_evidence_tests for token in (
                "test_world_state_v4_exposes_replayable_content_free_meta_governance",
                "test_legacy_world_state_v3_count_only_meta_remains_replayable",
                'record["changes"][0]["proposed_delta"] = 9.0',
                'match="phase schema version mismatch"',
            ))
            and all(token in manager_world_thread_tests for token in (
                "chronological == display_order",
                'chronological["meta_learning_authorized"] == 1',
            ))
            and all(token in meta_learning_tests for token in (
                'assert "matched_rows" not in encoded',
                'assert record["parameter_authority_granted"] is True',
                'assert record["authority_state"] == "denied"',
                'assert record["authority_state"] == "expired"',
                'assert record["evaluation"]["available"] is False',
            ))
            and all(token in web_tests for token in (
                'governance_renderer = document.split(',
                'assert ".innerHTML" not in governance_renderer',
                'assert "createElement(\'button\')" not in governance_renderer',
            ))
        ),
        "manager_product_exposes_one_replayable_world_evolution_thread": (
            all(token in manager_world_thread for token in (
                "def build_manager_world_evolution_thread(",
                "def validate_manager_world_evolution_thread(",
                "def manager_world_evolution_summary(",
                '"prematch_future_review"',
                'review_intent=terminal_review.get("intent")',
                'evidence_level=terminal_review.get("evidence_level")',
                'action_divergence_scenarios=int(',
                'local_attribution_scenarios=int(',
                'timing_sensitivity_observed=terminal_review.get(',
                '"frozen_manager_decision"',
                '"official_tactical_runtime"',
                '"official_world_model_actions"',
                '"observed_match_result"',
                '"persistent_world_state"',
                '"same_match_context_not_causal_direction"',
                '"review_to_official_world": review_world_certificate',
                '"reviewed_world_model_chain_complete"',
                '"scenario_to_runtime_opportunity_matching_performed": False',
                'review_world_certificate["certificate_identity"] = _identity(',
                "downstream_result_attribution_authorized=False",
                '"causal_effect_authorized": False',
                "manager world evolution thread replay mismatch",
            ))
            and all(token in decision_ledger for token in (
                "build_manager_world_evolution_thread(payload)",
                "validate_manager_world_evolution_thread(thread, entry=entry)",
                '"manager_world_evolution": manager_world_evolution_summary(entries)',
                '"evidence_level": (',
                '"fixed_scenario_budget": int(',
                '"descriptive_future_difference_scenarios": int(',
                '"source_scenario_identity": scenario.get("scenario_identity")',
                'archived["archive_identity"] = _identity(archived)',
                '"reviewed_scenarios": reviewed_scenarios',
            ))
            and all(token in web for token in (
                "function appendManagerWorldEvolutionThread(",
                "function renderManagerDecisionLedgerWorldEvolutionThread(season)",
                "function renderManagerIntelligenceWorldEvolutionThread(",
                "function appendManagerWorldEvolutionThreadReviewedFutures(",
                "function appendReviewWorldContinuity(",
                "function appendManagerWorldEvolutionThreadReviewWorldCertificate(",
                "stage.local_attribution_scenarios",
                "同场出现不等于因果",
            ))
            and "innerHTML" not in web
        ),
        "manager_world_evolution_render_pipeline_is_explicit": (
            all(token in web for token in (
                "function appendManagerWorldEvolutionThreadBase(host,thread)",
                "const MANAGER_WORLD_EVOLUTION_THREAD_STAGES=Object.freeze([",
                "appendManagerWorldEvolutionThreadSocietyContinuity",
                "appendManagerWorldEvolutionThreadReviewedFutures",
                "appendManagerWorldEvolutionThreadMechanismSemantics",
                "appendManagerWorldEvolutionThreadOfficialActionSemantics",
                "appendManagerWorldEvolutionThreadScenarioArchive",
                "appendManagerWorldEvolutionThreadReviewWorldCertificate",
                "appendManagerWorldEvolutionThreadRetainedRecordSemantics",
                "function appendManagerWorldEvolutionThread(host,thread){for(",
            ))
            and "appendManagerWorldEvolutionThreadWithout" not in web
            and web.count(
                "function appendManagerWorldEvolutionThread(host,thread)"
            ) == 1
        ),
        "manager_intelligence_render_pipeline_is_explicit": (
            all(token in web for token in (
                "function renderManagerIntelligenceBase(command,configured)",
                "const MANAGER_INTELLIGENCE_RENDER_STAGES=Object.freeze([",
                "renderManagerIntelligenceClubSupport",
                "renderManagerIntelligenceTacticalBinding",
                "renderManagerIntelligenceOfficialActionExecution",
                "renderManagerIntelligenceWorldEvolutionThread",
                "function renderManagerIntelligence(command,configured){for(",
            ))
            and "renderManagerIntelligenceWithout" not in web
            and web.count(
                "function renderManagerIntelligence(command,configured)"
            ) == 1
        ),
        "manager_future_set_render_pipeline_is_explicit": (
            all(token in web for token in (
                "function renderManagerFutureSets(season,managed)",
                "const renderManagerFutureSetsBase=renderManagerFutureSets;",
                "function renderManagerFutureSetsReviewActions(",
                "function renderManagerFutureSetsScenarioEvidence(",
                "function renderManagerFutureSetsMechanismExamples(",
                "const MANAGER_FUTURE_SET_RENDER_STAGES=Object.freeze([",
                "renderManagerFutureSets=(season,managed)=>{for(",
            ))
            and "renderManagerFutureSetsWithout" not in web
            and web.count(
                "renderManagerFutureSets=(season,managed)=>{for("
            ) == 1
        ),
        "recruitment_window_configure_pipeline_is_explicit": (
            all(token in web for token in (
                "function configureRecruitmentWindow(season)",
                "const configureRecruitmentWindowBase=configureRecruitmentWindow;",
                "function configureRecruitmentWindowLifecycle(",
                "function configureRecruitmentWindowGlobalMarket(",
                "function configureRecruitmentWindowSportingPlan(",
                "const RECRUITMENT_WINDOW_CONFIGURE_STAGES=Object.freeze([",
                "configureRecruitmentWindow=season=>{for(",
            ))
            and "configureRecruitmentWindowWithout" not in web
            and web.count(
                "configureRecruitmentWindow=season=>{for("
            ) == 1
        ),
        "season_render_pipeline_is_explicit": (
            all(token in web for token in (
                "function renderSeason(season,configured)",
                "const renderSeasonBase=renderSeason;",
                "function renderSeasonCommandCenter(",
                "function renderSeasonClubTimeline(",
                "function renderSeasonManagerFutureSets(",
                "const SEASON_RENDER_STAGES=Object.freeze([",
                "renderSeason=(season,configured,history=[],historySummary={})=>{for(",
            ))
            and "renderSeasonWithout" not in web
            and web.count(
                "renderSeason=(season,configured,history=[],historySummary={})=>{for("
            ) == 1
        ),
        "library_render_pipeline_is_explicit": (
            all(token in web for token in (
                "function renderLibraryBase(library)",
                "function renderLibraryBranchIdentity(",
                "function renderLibraryUnifiedFuture(",
                "function renderLibraryForkSets(",
                "const LIBRARY_RENDER_STAGES=Object.freeze([",
                "function renderLibrary(library){for(",
            ))
            and "renderLibraryWithout" not in web
            and web.count("function renderLibrary(library)") == 1
        ),
        "future_mechanism_example_pipeline_is_explicit": (
            all(token in web for token in (
                "function appendFutureMechanismExamplesBase(context)",
                "function appendFutureMechanismExamplesCrossMetrics(context)",
                "function appendFutureMechanismExamplesPolicySemantics(context)",
                "const FUTURE_MECHANISM_EXAMPLE_RENDER_STAGES=Object.freeze([",
                "function appendFutureMechanismExamples(host,scenarios,title)",
                "const context={host,scenarios:scenarios||[],title,details:null}",
                "context.details=details",
                "appendStage(context)",
            ))
            and "appendFutureMechanismExamplesWithout" not in web
            and web.count(
                "function appendFutureMechanismExamples(host,scenarios,title)"
            ) == 1
        ),
        "manager_counterfactual_workbench_is_one_replayable_workflow": (
            all(token in manager_intervention_workspace for token in (
                "def build_manager_intervention_workspace(",
                "def validate_manager_intervention_workspace(",
                '"freeze_intervention"',
                '"generate_bounded_futures"',
                '"inspect_local_mechanism"',
                '"record_manager_review"',
                '"advance_official_world"',
                '"outcome_effect_estimate": None',
                '"causal_effect_authorized": False',
                "manager intervention workspace replay mismatch",
            ))
            and all(token in web for token in (
                "build_manager_intervention_workspace(",
                "validate_manager_intervention_workspace(",
                '"manager_intervention_workspace"',
                "function renderManagerInterventionWorkspace(",
                "经理反事实干预五步流程",
                "从已验证进度恢复未来生成",
                "不排名时点、不预测比分",
            ))
            and "innerHTML" not in web
        ),
        "manager_season_has_one_current_and_historical_world_navigator": (
            all(token in manager_world_navigator for token in (
                "def build_manager_world_navigator(",
                "def validate_manager_world_navigator(",
                '"current_chapter"',
                '"history_chapters"',
                '"primary_action"',
                '"secondary_actions"',
                "all_chapters = sorted(",
                "(_history_chapter(row) for row in completed_entries)",
                '"visible_official_runtime_chapters"',
                '"world_model_influence_path"',
                '"world_model_action_adoption_ledger"',
                '"action_adoption"',
                "def _descriptive_world_propagation(",
                "def _season_world_trajectory(",
                '"world_trajectory": world_trajectory',
                '"reviewed_future_context"',
                '"future_to_outcome_comparison_performed": False',
                '"reviewed_future_scenario_evidence"',
                '"reviewed_future_action_divergence"',
                '"reviewed_future_local_attribution"',
                '"reviewed_future_timing_sensitivity"',
                '"complete_reviewed_world_model_chains"',
                '"review_to_official_world": copy.deepcopy(review_world)',
                "review-world certificate is invalid",
                "def _validate_reviewed_scenarios(",
                '"reviewed_scenario_archives"',
                '"identity_verified_scenario_archives"',
                '"scenarios": copy.deepcopy(reviewed_scenarios)',
                '"trajectory_point_identity"',
                '"world_change_markers"',
                '"descriptive_chronology_only": True',
                '"missing_evidence_imputed": False',
                '"turning_point_inference_authorized": False',
                "completed matchdays are duplicate",
                '"descriptive_world_after"',
                '"descriptive_cooccurrence_only": True',
                '"state_effect_comparison_authorized": False',
                "result_source_identity != _identity(observed)",
                'metrics_delta != source_match_delta.get("metrics_delta")',
                '"realized_change_rate_among_influenced"',
                '"direct_ball_event_link_coverage"',
                '"state_counts"',
                '"continuity_gap_counts"',
                '"latest_fixture_id"',
                '"latest_chapter_identity"',
                "chapter_key > current_key",
                "len(gaps) != len(set(gaps))",
                '"outcome_effect_estimate": None',
                '"causal_effect_authorized": False',
                "fixture identities are invalid or duplicate",
                "current continuity gaps are invalid",
                "manager world navigator replay mismatch",
            ))
            and all(token in web for token in (
                "build_manager_world_navigator(season)",
                "validate_manager_world_navigator(",
                '"manager_world_navigator"',
                'id="manager-world-navigator"',
                'id="manager-world-influence-path"',
                'id="manager-world-action-adoption-metrics"',
                'id="manager-world-action-adoption-diagnostics"',
                'id="manager-world-action-adoption-states"',
                'id="manager-world-action-propagation-boundary"',
                'id="manager-world-trajectory"',
                'id="manager-world-trajectory-list"',
                'id="manager-world-trajectory-boundary"',
                'id="manager-world-future-continuity-boundary"',
                'id="manager-world-gap-diagnostics"',
                "function renderManagerWorldNavigator(",
                "function renderManagerWorldActionAdoptionLedger(",
                "function managerWorldPropagationText(",
                "renderManagerWorldActionAdoptionLedgerWithoutWorldPropagation",
                "summary.world_model_influence_path||{}",
                "summary.world_model_action_adoption_ledger",
                "diagnosticRows=gapRows.concat(stateRows,trajectoryRows)",
                "function renderManagerWorldTrajectory(season)",
                "function renderManagerWorldReviewedFutureContinuity(season)",
                "future.action_divergence_scenarios",
                "row.fixture_id===fixtureId&&row.chapter_identity===expectedChapterIdentity",
                "summary.continuity_gap_counts||[]",
                "各项是独立证据覆盖，不是递减漏斗",
                "检查比赛是否启用世界模型并保留动作证据身份",
                "function visibleManagerLedgerEntries(",
                "node.dataset.fixtureId",
                "node.dataset.chapterComplete",
                "打开完整世界章节",
                "navigator.secondary_actions||[]",
                "function managerNavigationTarget(",
                "function openManagerWorldChapter(",
                "function navigateManagerWorldChapter(",
                "row.latest_fixture_id===fixtureId&&row.latest_chapter_identity===expectedChapterIdentity",
                "打开最近受影响章节",
                "history.pushState({seasonId,worldChapter:fixtureId}",
                "renderManagerDecisionLedger(currentSeason);renderManagerWorldNavigator(currentSeason)",
                "世界章节链接格式无效",
                "该世界章节链接属于另一个赛季",
                "当前导航目标不可用",
                "已完成的足球世界章节",
                "章节完整不代表赛果改善",
            ))
            and "innerHTML" not in web
        ),
        "historical_navigator_preserves_v2_action_semantics": (
            all(token in manager_world_navigator for token in (
                "_FUTURE_MECHANISM_SEMANTIC_FIELDS = (",
                '"cross_action_mechanism_examples"',
                '"direct_preference_mechanism_examples"',
                '"suppression_only_mechanism_examples"',
                '"hold_reference_redistribution_examples"',
                '"nonzero_cross_descriptive_windows"',
                "review_semantic_counts != terminal_semantic_counts",
                'review_stage.get(field, 0)',
                'terminal_review.get(field, 0)',
                '"reviewed_future_cross_actions"',
                '"reviewed_future_direct_preferences"',
                '"reviewed_future_suppression_only_signals"',
                '"reviewed_future_hold_reference_redistributions"',
                '"reviewed_future_nonzero_cross_windows"',
            ))
            and all(token in web for token in (
                "function renderManagerWorldReviewedFutureMechanismSemantics(season)",
                "trajectory.cross_action_mechanism_examples",
                "future.direct_preference_mechanism_examples",
                "future.suppression_only_mechanism_examples",
                "future.hold_reference_redistribution_examples",
                "future.nonzero_cross_descriptive_windows",
            ))
        ),
        "official_action_semantics_are_bounded_not_full_distribution": (
            all(token in manager_world_thread for token in (
                "def _official_action_semantic_examples(",
                '"retained_semantic_examples"',
                '"semantic_examples_truncated"',
                '"semantic_example_coverage_complete"',
                '"semantic_cross_action_examples"',
                '"semantic_direct_preference_examples"',
                '"semantic_suppression_only_examples"',
                '"semantic_hold_reference_redistribution_examples"',
                '"semantic_direct_cross_ball_event_examples"',
            ))
            and all(token in manager_world_navigator for token in (
                "_OFFICIAL_ACTION_SEMANTIC_COUNT_FIELDS = (",
                '"bounded_official_action_semantics"',
                '"bounded_semantic_examples"',
                '"fixtures_with_complete_coverage"',
                '"fixtures_with_truncated_examples"',
                '"full_record_distribution_authorized": False',
                "retained bounded official-action examples only; counts are ",
                "not a full-record action or signal distribution when any ",
            ))
            and all(token in web for token in (
                "function appendManagerWorldEvolutionThreadOfficialActionSemantics(",
                "renderManagerWorldActionAdoptionLedgerWithoutBoundedSemantics",
                "function renderManagerWorldReviewedFutureOfficialActionSemantics(season)",
                "sample.semantic_examples_truncated",
                "point.bounded_official_action_semantics",
            ))
        ),
        "official_action_v2_preserves_full_retained_record_semantics": (
            all(token in official_action_execution for token in (
                "SCHEMA_VERSION = 4",
                "V2_SCHEMA_VERSION = 2",
                "LEGACY_SCHEMA_VERSION = 1",
                "def _retained_record_semantics(",
                "def _validate_retained_record_semantics(",
                '"retained_record_semantics": _retained_record_semantics(',
                'elif "retained_record_semantics" in evidence:',
                '"full_source_distribution_authorized"',
                '"outcome_attribution_authorized": False',
            ))
            and all(token in decision_ledger for token in (
                'evidence["retained_record_semantics"]',
                '"fixtures_with_v2_semantics"',
                '"retained_record_semantics": semantic_summary',
            ))
            and all(token in manager_world_thread for token in (
                'action.get("retained_record_semantics")',
                "retained_record_semantics=retained_record_semantics",
            ))
            and all(token in manager_world_navigator for token in (
                "def _normalize_retained_record_semantics(",
                '"official_retained_record_semantics"',
                '"retained_record_semantics": retained_record_semantics',
                '"fixtures_without_v2_semantics"',
                '"all_chapters_have_v2_semantics"',
            ))
            and all(token in web for token in (
                "function retainedActionSemanticText(",
                "appendOfficialActionExecutionWithoutRetainedRecordSemantics",
                "function renderManagerDecisionLedgerRetainedRecordSemantics(season)",
                "function appendManagerWorldEvolutionThreadRetainedRecordSemantics(",
                "function renderManagerWorldRetainedRecordSemantics(season)",
                "point.official_retained_record_semantics",
            ))
        ),
        "official_action_v3_preserves_counterfactual_transition_matrix": (
            all(token in official_action_execution for token in (
                '"counterfactual_action_transition_counts"',
                '"locally_attributable_action_transition_counts"',
                "world-model retained action transition shape is invalid",
                "world-model retained action transitions are invalid",
                "V3_SCHEMA_VERSION = 3",
                "V3_SCHEMA_VERSION: 2",
            ))
            and all(token in decision_ledger for token in (
                '"fixtures_with_v3_transition_semantics"',
                '"all_official_evidence_has_v3_transition_semantics"',
                '"full_source_transition_distribution_authorized"',
            ))
            and all(token in manager_world_navigator for token in (
                "transition_semantic_rows = [",
                '"counterfactual_action_transition_counts": {',
                '"locally_attributable_action_transition_counts": {',
                '"all_chapters_have_v3_transition_semantics"',
                '"full_source_transition_distribution_authorized"',
            ))
            and all(token in web for token in (
                "retainedActionSemanticTextWithoutTransitions",
                "semantic?.locally_attributable_action_transition_counts",
                "semantic.retained_records_with_v2_semantics",
                "semantic.fixtures_with_v3_transition_semantics",
                "ledger.fixtures_without_v3_transition_semantics",
                "世界模型局部动作转移",
            ))
        ),
        "local_action_transitions_have_noncausal_world_propagation_ledger": (
            all(token in manager_world_navigator for token in (
                "def _local_transition_descriptive_propagation(",
                '"chapters_with_other_local_transitions"',
                '"chapter_membership_mutually_exclusive": False',
                '"cross_stratum_comparison_authorized": False',
                '"local_transition_descriptive_propagation": (',
                "chapters may enter multiple strata; no transition ",
                "effect, ranking, score causality or real-football claim",
            ))
            and all(token in web for token in (
                "function renderManagerWorldTransitionPropagation(",
                "local_transition_descriptive_propagation",
                "renderManagerWorldTransitionPropagation,",
                "row.chapters_with_other_local_transitions",
                "分层可重叠",
                "不排名、不比较效果、不归因赛果",
            ))
        ),
        "manager_action_transition_matrix_is_accessible_and_honest": (
            all(token in web for token in (
                'id="manager-world-action-transition-map"',
                'id="manager-world-action-transition-map-summary"',
                'id="manager-world-action-transition-table"',
                'id="manager-world-action-transition-table-body"',
                'role="region" aria-labelledby="manager-world-action-transition-map-title"',
                "function renderManagerWorldActionTransitionMap(season)",
                "semantic?.locally_attributable_action_transition_counts",
                "semantic?.fixtures_without_v3_transition_semantics",
                "managerWorldActionTransitionTableBody.replaceChildren()",
                "managerWorldActionTransitionTable.hidden=true",
                "这里显示的是证据缺口，不是“世界模型没有改变动作”",
                "矩阵格是计数，不是效果值",
                "renderManagerWorldActionTransitionMap,",
                ".transition-map-scroll { max-width:100%; overflow-x:auto",
            ))
        ),
        "action_change_expectation_matches_shared_uniform_sampler": (
            all(token in action_adoption_controller for token in (
                "def shared_uniform_action_change_probability(",
                "shared_uniform_inverse_cdf_overlap_v1",
                '"shared_uniform_change_probability": shared_uniform_change',
                'store["expected_counterfactual_action_changes"] += shared_uniform_change',
                'store["pass_target_expected_changes"] += shared_uniform_change',
                '"total_variation_distance": total_variation',
                '"expected_change_estimator": _EXPECTED_CHANGE_ESTIMATOR',
            ))
            and all(token in workspace for token in (
                '"expected_change_estimator": str(',
                'value.get("expected_change_estimator") or ""',
            ))
            and all(token in web for token in (
                "adoption.expected_change_estimator==='shared_uniform_inverse_cdf_overlap_v1'",
                "共享采样期望改变",
                "旧版期望改变",
            ))
        ),
        "official_action_v4_preserves_exact_shared_uniform_expectation": (
            all(token in official_action_execution for token in (
                "SCHEMA_VERSION = 4",
                "V3_SCHEMA_VERSION = 3",
                "evidence_schema_version = SCHEMA_VERSION if exact_source else V3_SCHEMA_VERSION",
                '"shared_uniform_change_probability"',
                '"expected_change_estimator": _EXPECTED_CHANGE_ESTIMATOR',
                '"expected_counterfactual_action_changes": round(sum(',
                '"full_source_expectation_authorized"',
                "world-model retained action expectation is invalid",
                "legacy official action evidence has V4 expectation fields",
            ))
            and all(token in decision_ledger for token in (
                '"fixtures_with_v4_expectation_semantics"',
                '"all_official_evidence_has_v4_expectation_semantics"',
                '"full_source_expectation_authorized"',
            ))
            and all(token in manager_world_thread for token in (
                'action.get("schema_version") in {2, 3, 4}',
                "retained_record_semantics=retained_record_semantics",
            ))
            and all(token in manager_world_navigator for token in (
                "semantic_schema_version not in {1, 2, 3}",
                '"fixtures_with_v4_expectation_semantics"',
                '"all_chapters_have_v4_expectation_semantics"',
                '"full_source_expectation_authorized"',
                "manager world navigator retained action expectation is invalid",
            ))
            and all(token in web for token in (
                "retainedActionSemanticTextWithoutExactExpectation",
                "function renderManagerDecisionLedgerExactActionExpectation(season)",
                "function renderManagerWorldExactActionExpectation(season)",
                "V4共享采样期望不可用",
                "不能解释为零影响",
            ))
        ),
        "action_transition_map_drills_into_descriptive_world_propagation": (
            all(token in manager_world_navigator for token in (
                "Stratify same-chapter facts by V3-compatible local action transition.",
                'in {2, 3}',
                '"local_transition_descriptive_propagation": (',
                '"cross_stratum_comparison_authorized": False',
                '"outcome_attribution_authorized": False',
            ))
            and all(token in web for token in (
                'id="manager-world-action-transition-detail"',
                'id="manager-world-action-transition-open"',
                "function renderManagerWorldActionTransitionDrilldown(season)",
                "new Map(rows.map(row=>[row.transition_id,row]))",
                "button.className='transition-cell-button'",
                "button.setAttribute('aria-controls','manager-world-action-transition-detail')",
                "button.setAttribute('aria-pressed','false')",
                "managerWorldActionTransitionDetail.focus()",
                "不排名、不比较跨格效果、不归因赛果",
            ))
        ),
        "all_known_exposed_credentials_require_independent_v2_closure": (
            security_protocol.get("schema_version") == 2
            and [
                row.get("incident_id")
                for row in security_protocol.get("incidents", [])
            ] == ["deepseek_api_credential", "github_classic_pat"]
            and security_protocol.get("evidence_contract", {}).get(
                "distinct_receipt_per_incident_required"
            ) is True
            and security_protocol.get("evidence_contract", {}).get(
                "distinct_receipt_content_required"
            ) is True
            and security_protocol.get("evidence_contract", {}).get(
                "maximum_receipt_bytes"
            ) == 10 * 1024 * 1024
            and security_protocol.get("evidence_contract", {}).get(
                "chronological_timestamps_required"
            ) is True
            and security_protocol.get("evidence_contract", {}).get(
                "secret_pattern_version"
            ) == "gfs_known_credentials_v2"
            and security_protocol.get("evidence_contract", {}).get(
                "origin_remote_credential_forbidden"
            ) is True
            and all(token in security_verifier for token in (
                '"github_classic_pat": re.compile(',
                '"github_fine_grained_pat": re.compile(',
                '"all_known_incidents_are_present_exactly_once"',
                '"every_redacted_receipt_is_distinct_confined_and_content_addressed"',
                "len(set(receipt_digests)) == len(KNOWN_INCIDENTS)",
                "_receipt_size_is_bounded(",
                '"revocation_and_signature_timestamps_are_chronological"',
                '"known_incidents_complete": passed',
                '"origin_remote_contains_no_embedded_credential"',
            ))
            and all(token in completion_plan for token in (
                "security_closure_v2/attestation.json",
                "security_closure_verification_v2.json",
                "distinct redacted DeepSeek and GitHub revocation receipts",
            ))
            and all(token in control_plane for token in (
                "security_closure_protocol_verification_v2.json",
                "security_closure_verification_v2.json",
                'security_closure.get("known_incidents_complete") is True',
                'security_protocol.get("origin_has_embedded_credential") is False',
            ))
            and all(token in evidence_kit for token in (
                '"security": "data/evaluation/security_closure_protocol_v2.json"',
                "deepseek_api_credential_receipt.template.json",
                "github_classic_pat_receipt.template.json",
            ))
        ),
        "manager_advice_preview_is_confidence_aware_and_non_causal": (
            all(token in decision_advice for token in (
                "def build_manager_advice_comparison(",
                "def validate_manager_advice_comparison(",
                '"exploratory_only" if reasons else "bounded_review"',
                '"automatic_adoption_authorized": False',
                '"performance_validated": False',
                '"recommended_minus_selected": {',
                "world-model advice comparison replay mismatch",
            ))
            and all(token in workspace for token in (
                "build_manager_advice_comparison,",
                '"comparison": build_manager_advice_comparison(',
                "advice, selected_tactic=frozen[\"tactic\"]",
            ))
            and all(token in web for token in (
                "renderManagerDecisionPreviewWithoutWorldModelComparison",
                "authority.level==='exploratory_only'",
                "comparison.recommended_tactic",
                "comparison.selected_tactic",
                "deltas.risk_adjusted_value",
                "adoptManagerAdvice.textContent=",
            ))
            and "innerHTML" not in web
        ),
        "cross_action_authority_is_distinct_replayable_and_fail_closed": (
            all(token in cross_validation for token in (
                "grouped_heldout_simulator_cross_transitions",
                "same_state_persistence",
                "CROSS_MIN_SAMPLES = 96",
                "CROSS_MIN_GROUPS = 6",
                "CROSS_MIN_SKILL = 0.02",
                "replay_cross_action_validation",
            ))
            and all(token in wm_inference for token in (
                "self.cross_validation = replay_cross_action_validation(",
                "self.cross_quality = float(self.cross_validation",
                'elif kind == "cross":',
                "quality = self.cross_quality",
                'quality_kind="cross"',
            ))
            and all(token in world_model_planner for token in (
                "validated_cross_vs_continuation_advantage",
                "cross_quality_gate_closed",
                "runtime.score_cross_action(",
            ))
            and "def record_cross(" in ball_path_logger
            and '"type": "cross"' in ball_path_logger
            and 'if et == "cross":' in action_codec
            and '"pass", "shot", "cross"' in product_replay
            and "cross_action_validation = build_cross_action_validation(" in trainer
            and cross_protocol.get("state") == "registered_validation_contract"
            and cross_protocol.get("validation", {}).get("external_football_validity") is False
            and cross_verification.get("status")
            == "blocked_checkpoint_missing_cross_validation"
            and cross_verification.get("code_ready") is True
            and set(
                (cross_verification.get("execution_identity") or {})
                .get("code_sha256", {})
            ) == set(cross_protocol.get("required_code_paths") or [])
            and cross_verification.get("cross_planning_authorized") is False
            and cross_verification.get("runtime_cross_quality") == 0.0
            and cross_verification.get("training_executed") is False
            and cross_verification.get("matches_executed") == 0
            and cross_verification.get("provider_calls_made") is False
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
        "formal_outcome_ledger_is_fixed_authorized_and_input_bound": (
            all(token in formal_runner for token in (
                'AUTHORIZATION = "I_AUTHORIZE_GFS_FORMAL_EXPERIMENT_V2"',
                "def _require_formal_authorization(",
                "def validate_progress(",
                '"rows_follow_exact_schedule_prefix"',
                '"calibration_and_action_metrics_are_finite"',
                '"action_counts_are_bounded"',
                '"baseline_has_zero_action_activity"',
                '"arm_order_is_resumable"',
                '"blocked_invalid_progress"',
                '"analysis_input_sha256"',
                'candidate_rows = [*arm_state["rows"], dict(row)]',
                'if not validation["structural_valid"]',
                "require_complete=True",
                "_require_valid_progress(validate_progress(",
            ))
            and formal_runner.count(
                "_require_valid_progress(validate_progress("
            ) >= 3
        ),
        "runtime_random_draws_are_identity_scoped": (
            not global_rng_draw_violations
            and "def named_py_rng(" in (
                ROOT / "src/simulation/random_control.py"
            ).read_text(encoding="utf-8")
            and all(token in (
                ROOT / "src/simulation/tournament_match.py"
            ).read_text(encoding="utf-8") for token in (
                "match_seed = derive_seed(", '"tournament_match"',
            ))
            and 'named_py_rng(root_seed, "legacy_journey", team)' in (
                ROOT / "src/memory_engine/tournament_simulator.py"
            ).read_text(encoding="utf-8")
            and all(token in (
                ROOT / "src/simulation/engine.py"
            ).read_text(encoding="utf-8") for token in (
                '"active_agents"', '"agent_action"', '"headline_match"',
            ))
        ),
        "tournament_resume_binds_random_world_identity": (
            all(token in tournament_checkpoint for token in (
                "CHECKPOINT_VERSION = 6",
                'RANDOM_WORLD_CONTRACT = "identity_scoped_rng_v1"',
                "def _content_sha256(",
                "Tournament checkpoint content integrity mismatch",
                "def checkpoint_root_seed(",
                "Legacy tournament checkpoint V1 lacks random-world identity",
            ))
            and all(token in public_app for token in (
                "def _resolve_tournament_root_seed(",
                "Explicit tournament seed conflicts with checkpoint root seed",
                "initialization_seed=seed",
                "build_simulation(root, seed=seed)",
                "config = SimulationConfig.from_mapping(runtime_values)",
            ))
            and all(token in tournament_runtime for token in (
                'self.root_seed = int(getattr(world_engine, "root_seed", 42))',
                "base_dir=None",
            ))
            and all(token in tournament_state for token in (
                "root_seed=self.root_seed",
                "stored_seed = checkpoint_root_seed(ckpt)",
                "does not match the current world",
            ))
            and all(token in world_runner for token in (
                "TournamentManager(", "engine, base_dir=base_dir",
                "run_identity_sha256=run_identity_sha256",
            ))
        ),
        "tournament_resume_binds_portable_run_input_identity": (
            all(token in runtime for token in (
                "def _source_tree_identity(",
                '"scope": "src_python_tree_v1"',
                "def safe_runtime_options(",
                "def _safe_config_payload(",
                "_SENSITIVE_OPTION_FRAGMENTS",
                "def run_manifest_identity(",
                '"schema_version": 2',
                'manifest["run_identity_sha256"] = run_manifest_identity(manifest)',
                'options[key] = "sha256:" + hashlib.sha256(value).hexdigest()',
            ))
            and all(token in tournament_checkpoint for token in (
                "def checkpoint_run_identity(",
                "def capture_state_artifacts(",
                "def verify_state_artifacts(",
                "external state integrity mismatch",
                '"run_identity_sha256": run_identity_sha256',
                "Tournament checkpoint V2 lacks full run identity",
            ))
            and all(token in public_app for token in (
                "def _verify_tournament_resume_identity(",
                "code, data, model, or configuration identity drift",
                'runtime_values=runtime_values',
                'run_identity_sha256=manifest["run_identity_sha256"]',
                'runtime_values.get("MATCH_WM_CHECKPOINT"',
                'runtime_values.get("MATCH_WM_SHOT_HEAD"',
                "def _tournament_input_paths(",
                'root / "data" / "rosters"',
            ))
            and all(token in tournament_state for token in (
                "self._require_run_identity()",
                "def _require_run_identity(",
                "checkpoint_run_identity(ckpt) != self.run_identity_sha256",
            ))
        ),
        "tournament_resume_restores_dynamic_world_and_receipted_reflection": (
            all(token in tournament_world_state for token in (
                "WORLD_STATE_VERSION = 1",
                "AGENT_REQUIRED_FIELDS",
                "AGENT_REBUILT_FIELDS",
                "Unclassified mutable Agent fields",
                "def snapshot_world_state(",
                "def restore_world_state(",
                "Validate every live identity before mutating any in-memory world object",
                "manager.dialogue_engine.topic_market",
                "manager.narrative_event_bus.history",
            ))
            and all(token in tournament_checkpoint for token in (
                "CHECKPOINT_VERSION = 6",
                '"world_state": validate_world_state(world_state)',
                '"reflection_journal": validate_reflection_journal(reflection_journal)',
                "def validate_reflection_world_consistency(",
                "Applied reflection is missing from Agent state",
                "def _directory_sha256(",
                '"data/persistence/world_model_fusion.jsonl"',
                '"data/persistence/tactical_counterfactuals.jsonl"',
                "Tournament checkpoint V3 lacks dynamic world state",
            ))
            and all(token in tournament_state for token in (
                "world_state=snapshot_world_state(self)",
                'ckpt["world_state"]',
            ))
            and all(token in tournament_lifecycle for token in (
                "Persist the provider result before it can mutate the world",
                "agent.request_reflection_payload(llm)",
                'journal["applied"].append(operation_id)',
                "def _record_final_result(",
            ))
            and all(token in agent_reflection for token in (
                "def request_reflection_payload(",
                "def apply_reflection_payload(",
                'record.get("operation_id") == operation_id',
                "Reflection response must be a JSON object",
            ))
            and all(token in roster_loader for token in (
                "def _sanitize_roster_numerics(",
                "Treat legacy NaN/Infinity observations as absent",
                "def _finite(",
            ))
            and "math.isfinite(float(dynamics[key]))" in world_runner
        ),
        "tournament_resume_rolls_back_internal_external_state_after_identity": (
            all(token in tournament_checkpoint for token in (
                "STATE_SNAPSHOT_VERSION = 1",
                "MAX_STATE_SNAPSHOT_BYTES",
                "MAX_STATE_SNAPSHOT_FILES",
                "def capture_state_snapshot(",
                "def validate_state_snapshot(",
                "def restore_state_artifacts(",
                "Tournament state snapshot file integrity mismatch",
                "Invalid tournament state snapshot path",
                "Automatic recovery cannot modify an external cognitive cache",
                "with FileLease(lock_path, timeout=5.0)",
                "Tournament checkpoint V4 lacks recoverable external state",
                '"state_snapshot": capture_state_snapshot(base_dir, state_artifacts)',
                "verify_external_state: bool = True",
            ))
            and all(token in public_app for token in (
                "def _recover_tournament_external_state(",
                "verify_external_state=False",
                "Tournament recovery requires an identity-matched checkpoint",
                "_verify_tournament_resume_identity(",
                "_recover_tournament_external_state(",
                "External rollback mutates files",
                "checkpoint=checkpoint",
            ))
        ),
        "tournament_match_is_one_verified_rollback_complete_transaction": (
            all(token in tournament_checkpoint for token in (
                "CHECKPOINT_VERSION = 6",
                "STATE_ARTIFACT_DIRECTORIES = (",
                '"data/persistence/cognitive_log"',
                '"outputs/ball_log"',
                '"outputs/narrative_debug.jsonl"',
                "def require_internal_match_transaction_targets(",
                "Tournament checkpoint V5 lacks transactional match-output state",
            ))
            and all(token in tournament_match for token in (
                "with tournament_match_transaction(",
                "return self._play_match_once(",
            ))
            and all(token in tournament_transaction for token in (
                "class TournamentMatchRollbackError(RuntimeError):",
                "@contextmanager",
                "with FileLease(lock_path, timeout=0.0) as lease:",
                "manager._save_checkpoint()",
                "checkpoint = load_checkpoint(manager.base_dir)",
                "def _verify_match_commit(",
                "restore_state_artifacts(manager.base_dir, checkpoint)",
                "manager._restore_from_checkpoint(checkpoint)",
                "def _verify_rollback(",
                "snapshot_world_state(manager)",
                "Tournament in-memory rollback did not converge",
                "Tournament durable checkpoint rollback did not converge",
                '"contains_error_message": False',
            ))
        ),
        "public_tournament_lifecycle_shares_the_match_workspace_lease": (
            all(token in public_app for token in (
                "from src.simulation.tournament_transaction import tournament_workspace_lease",
                "with tournament_workspace_lease(root):",
                "def _run_full_tournament_locked(",
            ))
            and all(token in tournament_transaction for token in (
                "_ACTIVE_WORKSPACE_LEASE: ContextVar[",
                "def tournament_workspace_lease(",
                "active_path != lock_path or not lease.held",
                "with FileLease(lock_path, timeout=0.0) as lease:",
                "with tournament_workspace_lease(manager.base_dir):",
            ))
        ),
        "production_validation_is_deployable_serialized_and_identity_bound": (
            all(token in production_validation for token in (
                'persistence / "product_web.lock"',
                'persistence / "production_validation.lock"',
                "on_task_claimed=on_task_claimed",
                '"production_validation_task_running"',
                "recovered_identity_replayed",
                '"idempotency_sha256"',
                '"progress_sha256": _payload_sha256(progress)',
                "final_progress_snapshot_is_bound",
                "restore scratch must be outside the validation workspace",
                "def record_attestation(",
                "def finalize(",
            ))
            and "COPY scripts/run_production_validation.py" in dockerfile
            and all(token in compose for token in (
                "production-validation:",
                'profiles: ["validation"]',
                'network_mode: "none"',
                "gfs_validation_persistence:/app/data/persistence",
                "gfs_validation_evaluation:/app/data/evaluation/production_validation_v1",
                "gfs_validation_scratch:/validation-scratch",
            ))
        ),
        "human_studies_are_registered_allocated_and_byte_verified": (
            all(token in human_study for token in (
                "with FileLease(",
                'timeout=5.0',
                '"registered_at": registered_at',
                '"case_pack_manifest_sha256": case_pack_manifest_sha256',
                "def bind_records_to_registry(",
                "def verify_content_addressed_archive(",
                'candidate.is_symlink() or not candidate.is_file()',
                'hasher = hashlib.sha256()',
                'hasher.hexdigest() != digest',
                'retained evidence bytes require a positive size limit',
                '"verified_bytes": verified_bytes',
            ))
            and all(token in product_validation_study for token in (
                'parser.add_argument("--register", action="store_true")',
                "registry_binding = bind_records_to_registry",
                "archive_report = verify_content_addressed_archive",
                '"session_registry": _file_sha256(registry_path)',
            ))
            and all(token in product_value_study for token in (
                "def validate_case_packs(",
                "derived_winners.append",
                "def _score_receipt(",
                'archive_report["verified_bytes"]',
                "max_bytes_per_artifact=case_manifest",
                "declared correctness disagrees with frozen scoring",
                "case_pack_manifest_path=case_pack_path",
            ))
            and product_validation_protocol.get("execution", {}).get(
                "participants_observed"
            ) == 0
            and product_value_protocol.get("current_execution", {}).get(
                "participants_observed"
            ) == 0
            and product_value_cases.get("current_execution", {}).get(
                "participants_exposed"
            ) == 0
        ),
        "product_value_delivery_is_blinded_scoring_sealed_and_product_visible": (
            all(token in study_delivery for token in (
                "class ProductValueStudyDelivery:",
                "def status(self)",
                "def register(",
                "def packet(",
                "def materialize_packet(",
                "participant packet contains forbidden fields",
                "_reject_existing_symlink_components",
                "os.link(temporary, target)",
                '"scoring_material_included": False',
                '"moderator_identity_included": False',
            ))
            and all(token in cli for token in (
                "p_value_study = studio_sub.add_parser(",
                "value_study_sub = p_value_study.add_subparsers(",
                "p_value_status = value_study_sub.add_parser(",
                "p_value_register = value_study_sub.add_parser(",
                "p_value_packet = value_study_sub.add_parser(",
                "cmd_studio_value_study_packet",
            ))
            and all(token in web for token in (
                '"/api/v1/studies/product-value": self._product_value_study_status',
                '"/api/v1/studies/product-value/registrations": (',
                'r"/api/v1/studies/product-value/packets/',
                'id="product-value-study-panel"',
                "X-GFS-Blinded-Study-Packet",
            ))
            and all(token in evidence_kit for token in (
                "participant_case_authority_excludes_scoring_material",
                "moderator_scoring_seal_is_excluded_from_kit",
            ))
            and all(
                token not in json.dumps(product_value_cases)
                for token in ('"scoring_key"', '"scoring_keys"')
            )
            and product_value_cases.get("delivery_contract", {}).get(
                "participant_manifest_contains_scoring_keys"
            ) is False
            and product_value_protocol.get("integrity", {}).get(
                "participant_delivery_excludes_scoring_material"
            ) is True
            and product_value_protocol.get("integrity", {}).get(
                "scoring_seal_is_separate_and_case_hash_bound"
            ) is True
            and product_value_protocol.get("outputs", {}).get("scoring_seal")
            == "data/evaluation/product_value_scoring_seal_v1.json"
            and product_value_scoring_seal.get("case_pack_manifest_sha256")
            == file_sha256(
                ROOT / "data/evaluation/product_value_case_packs_v1.json"
            )
            and product_value_scoring_seal.get("access_contract")
            == {
                "participant_delivery_forbidden": True,
                "moderator_analysis_only": True,
                "repository_is_not_a_participant_delivery_channel": True,
            }
            and product_value_scoring_seal.get("current_execution")
            == {
                "participants_exposed": 0,
                "measured_sessions": 0,
                "results_available": False,
            }
        ),
        "product_value_participant_session_is_isolated_timed_and_attested": (
            all(token in study_session for token in (
                "class ParticipantSessionRuntime:",
                "class ParticipantStudyWebApp:",
                "def provision_participant_session(",
                "def import_completed_session(",
                '"capability_stored_in_plaintext": False',
                '"participant_process_requires_repository_access": False',
                '"future_condition_exposed": False',
                '"correctness_disclosed": False',
                "_advance_expired",
                "_ensure_completion_record",
                "with FileLease(self.lease_path, timeout=5.0)",
                "verify_content_addressed_archive(",
                "explicit moderator observer attestation is required",
                'path == "/api/v1/conditions/current/submit"',
                "script-src 'nonce-",
                "style-src 'nonce-",
                '"trusted_proxy_required"',
                '"idempotency_conflict"',
            ))
            and all(token in cli for token in (
                "cmd_studio_value_study_provision",
                "cmd_studio_value_study_serve",
                "cmd_studio_value_study_import",
                '"--attest-observed-session"',
            ))
            and product_value_protocol.get("session_execution")
            == {
                "runner_kind": "gfs_product_value_participant_session_v1",
                "participant_runtime_is_external_to_repository": True,
                "capability_token_minimum_characters": 32,
                "capability_token_minimum_distinct_characters": 8,
                "capability_plaintext_persistence_forbidden": True,
                "launch_token_transport": "url_fragment_then_bearer_header",
                "training_task_required_before_measurement": True,
                "training_task_is_unscored": True,
                "condition_content_revealed_only_after_server_start": True,
                "future_condition_preexposure_forbidden": True,
                "server_authoritative_deadline_seconds": 900,
                "submission_policy": (
                    "first_valid_submission_wins_idempotent_by_submission_id"
                ),
                "idempotent_retry_requires_exact_answers_and_verified_receipt": True,
                "completion_commit_policy": (
                    "atomic_state_first_then_idempotent_derived_record"
                ),
                "remote_transport_policy": (
                    "loopback_service_same_host_trusted_https_proxy"
                ),
                "correctness_feedback_during_session": False,
                "critical_error_policy": {
                    "field": "supported_claim",
                    "values": [
                        "proven_real_world_tactic", "guaranteed_match_win",
                    ],
                },
                "completed_record_import_requires_moderator_attestation": True,
                "participant_service_exposes_studio_routes": False,
            }
            and product_value_protocol.get("current_execution")
            == {
                "participants_observed": 0,
                "sessions_executed": 0,
                "matches_executed": 0,
                "training_executed": False,
                "provider_calls_made": False,
                "results_available": False,
            }
        ),
        "release_readiness_separates_code_contract_from_external_results": (
            all(token in control_plane for token in (
                "target_identity_current",
                '"target_lock_is_complete_and_hashed"',
                '"target_lock_validation_matches_artifacts"',
                "PAPER_PACKAGE_CHECKS = frozenset({",
                'PAPER_EXTERNAL_RESULT_CHECK = "formal_result_identity_and_replay_pass"',
                "paper_code_ready = (",
                "and set(paper_checks) == PAPER_PACKAGE_CHECKS",
                "and paper_checks.get(PAPER_EXTERNAL_RESULT_CHECK) is False",
                "if name != PAPER_EXTERNAL_RESULT_CHECK",
                "infrastructure_code_ready = all(",
                '"code_contract_checks": {',
                '"paper_package_without_confirmatory_result": paper_code_ready',
            ))
            and all(token in cli for token in (
                '"status": release["status"]',
                '"code_ready": release["code_ready"]',
                '"code_contract_checks": release["code_contract_checks"]',
                '"next_action": release["next_action"]',
            ))
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
        "unsealed_joint_shot_head_cannot_control_actions": (
            all(token in wm_inference for token in (
                "self.joint_shot_quality = float(",
                "self.shot_quality = 0.0",
                'self.shot_probability_source = "physics_xg_prior"',
                "goal_term = xg_prior",
                'self.planner_authority(obs, kind="shot")',
            ))
            and candidate.get("shot_planner_active") is False
            and candidate.get("shot_fallback") == "physics_xg_prior"
            and (candidate.get("sealed_test") or {}).get("shot", {}).get(
                "skill_vs_physics_xg_prior", 0.0
            ) < 0.0
            and all(token in workspace_evidence for token in (
                '"shot_action_validation": {',
                '"joint_head_authorized": False',
                '"physics_xg_fallback_joint_head_not_authorized"',
            ))
            and all(token in web for token in (
                "evidence?.shot_action_validation||{}",
                "shot.frozen_head_authorized",
                "shot.skill_vs_physics_xg_prior",
            ))
        ),
        "llm_pool_is_config_scoped_and_locked": (
            "dict[LLMGatewayConfig" in gateway and "_GATEWAY_LOCK" in gateway
        ),
        "llm_success_count_requires_accepted_content": (
            gateway.index("extract_json_object(content)", gateway.index("def complete("))
            < gateway.index("self._request_succeeded()", gateway.index("def complete("))
            and "self.call_count += 1" in gateway
        ),
        "llm_transport_is_scope_bound_and_evidence_bearing": (
            all(token in gateway for token in (
                "class ProviderAdapter(Protocol)",
                "X-GFS-Request-ID",
                "def request_scope(",
                "aggregate_successful_calls",
                "class LLMCircuitOpenError",
                "contains_prompts_or_credentials",
            ))
            and all(token in workspace for token in (
                "provider_transport_scope_mismatch",
                "provider_transport_call_count_mismatch",
                "provider_transport_secret_boundary_unverified",
                "scope_factory(match_id)",
            ))
        ),
        "squad_construction_is_root_bound_and_provenance_visible": (
            all(token in squad_factory for token in (
                "base_dir: str | Path | None = None",
                "load_effective_roster(base_dir, team_id)",
                "cannot form an available XI",
                'roster_source = "effective_roster"',
                '"source": roster_source',
                '"source": "synthetic_status_fallback"',
                '"roster_identity": roster_identity(roster)',
            ))
            and all(token in macro_bridge for token in (
                "base_dir: str | Path | None = None",
                "base_dir=base_dir",
            ))
            and all(token in match_micro_runner for token in (
                "base_dir=resolved_base_dir",
                "squad_provenance={",
                '"fallback_used": bool(',
            ))
            and all(token in workspace for token in (
                '"roster": raw.get("squad_provenance")',
                "research_synthetic_roster_fallback",
                "base_dir=self.root",
            ))
        ),
        "match_runtime_inputs_are_side_isolated_and_root_bound": (
            all(token in internal_signals for token in (
                "def normalize_internal_match_signals(",
                "coordination_away=_bounded_signal(",
                "internal_away,\n            \"coordination\"",
                "conflict_away=_bounded_signal(",
                "internal_away,\n            \"conflict_heat\"",
                "math.isfinite(value)",
                "MAX_CONFLICT_HEAT = 1.05",
            ))
            and all(token in match_micro_runner for token in (
                "normalize_internal_match_signals(",
                "conflict_away = internal_signals.conflict_away",
            ))
            and all(token in match_affective_runner for token in (
                "normalize_internal_match_signals(",
                "base_dir=base_dir",
            ))
            and match_pipeline.count(
                "base_dir: str | os.PathLike[str] | None = None"
            ) >= 3
            and match_pipeline.count("base_dir=base_dir") >= 3
            and tournament_scoring.count("base_dir=self.base_dir") >= 2
            and tournament_reporting.count("base_dir=self.base_dir") >= 2
        ),
        "physics_official_score_path_fails_closed": (
            all(token in score_path for token in (
                "class OfficialScoreIntegrityError(ValueError):",
                "def _official_goal(",
                "def _official_xg(",
                "def validate_official_xg_prior(",
                "def validate_physics_official_summary(",
                "Physics-official score cannot contain an xG supplement",
                "Physics-official goals do not match physics goal evidence",
            ))
            and all(token in tournament_scoring for token in (
                "class PhysicsOfficialScoreError(RuntimeError):",
                "Physics-official regulation failed before score commit",
                "Physics-official extra time failed before score commit",
                "finalize_official_score_from_micro(",
            ))
            and "falling back to macro score" not in tournament_scoring
            and "MATCH_MICRO_STRICT" not in tournament_scoring
            and all(token in workspace for token in (
                "validate_physics_official_summary(raw)",
                '"physics_evidence_validated": (',
                "physics_score_evidence is None",
            ))
            and all(token in product_reporting for token in (
                'data-testid="score-provenance"',
                "Physics score evidence verified",
                "Physics score evidence not verified",
            ))
        ),
        "prospective_m2_identity_closes_transitive_runtime_dependencies": (
            all(token in code_identity for token in (
                'TRANSITIVE_LOCAL_IMPORTS_V1 = "transitive_local_imports_v1"',
                "def code_identity_manifest(",
                "relative.is_absolute()",
                '".." in relative.parts',
                "ast.parse(",
                "for relative in sorted(resolved)",
            ))
            and "code_identity_manifest(" in mirrored_policy_evaluation
            and '"transitive_local_imports_v1"' in m2_study
            and m2_integrity.get("code_identity_mode")
            == "transitive_local_imports_v1"
            and (m2_integrity.get("identity_amendment") or {}).get(
                "formal_runs_before_amendment"
            ) == 0
            and m2_preflight.get("training_executed") is False
            and m2_preflight.get("checkpoint_written") is False
            and m2_preflight.get("evidence_identity")
            == m2_expected_preflight_identity
            and {
                "src/infrastructure/code_identity.py",
                "src/match_engine/match_micro_runner.py",
                "src/match_engine/internal_signals.py",
                "src/simulation/match_pipeline.py",
            }.issubset(m2_preflight_code)
        ),
        "m2_joint_changing_action_utility_is_trained_validated_and_gated": (
            all(token in m2_trainer for token in (
                'objective_scope="two_step_policy_utility"',
                'action_sequence="observed_changing_actions"',
                "policy_utility_two_step_optimization_steps += 1",
                '"policy_utility_two_step": two_step_policy_utility_validation',
            ))
            and all(token in m2_validator for token in (
                "policy_utility_sequence_validation_gate(",
                '"policy_utility_two_step": policy_utility_two_step',
                "development_policy_utility_sequence_gate.get",
                "sealed_policy_utility_sequence_gate.get",
            ))
            and "def policy_utility_sequence_authority(" in m2_runtime
            and all(token in m2_policy_utility for token in (
                "def policy_utility_sequence_validation_gate(",
                "def build_continuation_support_evidence(",
                "def continuation_support_validation_gate(",
                "def policy_utility_perspective_gate_integrity(",
                "def policy_utility_aggregate_gate_integrity(",
                'contract.get("action_sequence") == "observed_changing_actions"',
                'contract.get("trained_with_action_sequence_objective") is True',
            ))
            and all(token in m2_study for token in (
                '"policy_utility_two_step.pass"',
                (
                    '"policy_utility_two_step.pass.perspectives.home.'
                    'continuation_support"'
                ),
                (
                    '"policy_utility_two_step.pass.perspectives.away.'
                    'continuation_support"'
                ),
                '"required_policy_utility_sequence_gates"',
                '"sealed_pass_policy_utility_two_step_gate"',
            ))
            and (
                "two_step_policy_utility_sequence_objective_will_activate"
                in m2_preflight_source
            )
            and "development_pass_continuation_support_sufficient"
            in m2_preflight_source
            and "development_pass_perspective_support_sufficient"
            in m2_preflight_source
            and m2_preflight.get("checks", {}).get(
                "two_step_policy_utility_sequence_objective_will_activate"
            ) is True
            and (m2_integrity.get("identity_amendment") or {}).get("version") == 5
            and (m2_protocol.get("candidate") or {}).get(
                "required_sealed_validation"
            ) == [
                "two_step",
                "policy_utility.pass",
                "policy_utility.pass.perspectives.home",
                "policy_utility.pass.perspectives.away",
                "policy_utility_two_step.pass",
                (
                    "policy_utility_two_step.pass.perspectives.home."
                    "continuation_support"
                ),
                (
                    "policy_utility_two_step.pass.perspectives.away."
                    "continuation_support"
                ),
            ]
        ),
        "m2_runtime_consumes_supported_continuation_sequences": (
            all(token in m2_runtime for token in (
                "def predict_policy_utility_sequence(",
                "self.model.transition_rollout_predictions(",
                "strip_outcome_leakage(raw_actions)",
                '"explicit_changing_action_sequence_rollout"',
                '"open_loop_evidence_supported_continuation_policy"',
                "self.policy_utility_sequence_authority(",
            ))
            and all(token in world_model_planner for token in (
                'sequence_authority.get("continuation_policy")',
                "decode_action_kind(vector) == kind",
                '"action_vs_zero_persistence_reference"',
                '"exact_zero_persistence_reference"',
                '"policy_utility_sequence_runtime_contract_missing"',
                "sequence_authority_method(",
                "two_step_gate_method()",
                "sequence_policy = world_model_outcome_aligned_policy_enabled()",
                "sequence_policy=sequence_policy",
            ))
            and all(token in m2_study for token in (
                '"sequence_predictor_available"',
                "policy_utility_aggregate_gate_integrity(",
                "policy_utility_perspective_gate_integrity(",
                "def _perspective_policy_gate_ready(",
                "def _single_sequence_gate_ready(",
                "def _sequence_gate_ready(",
                'runtime, "predict_policy_utility_sequence", None',
                "and sequence_predictor_available",
            ))
            and (
                "def _receipt_sequence_gate_open("
                in mirrored_policy_evaluation
            )
            and (
                "def _receipt_policy_gate_open("
                in mirrored_policy_evaluation
            )
            and (
                "def _receipt_single_sequence_gate_open("
                in mirrored_policy_evaluation
            )
            and "policy_utility_aggregate_gate_integrity(" in (
                mirrored_policy_evaluation
            )
            and "policy_utility_perspective_gate_integrity(" in (
                mirrored_policy_evaluation
            )
            and "def _shared_pass_hold_continuation_policy(" not in world_model_planner
            and "def pass_imagination_bonuses(" not in world_model_planner
            and (m2_protocol.get("candidate") or {}).get(
                "continuation_support_contract", {}
            ).get("runtime_consumers") == [
                "high_level_pass_utility",
                "pass_target_ranking",
            ]
            and (m2_protocol.get("candidate") or {}).get(
                "continuation_support_contract", {}
            ).get("runtime_partition") == "attacking_home"
            and (m2_protocol.get("candidate") or {}).get(
                "continuation_support_contract", {}
            ).get("qualification_requires") == ["home", "away"]
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
            "direct_global_runtime_rng_draws": global_rng_draw_violations,
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
