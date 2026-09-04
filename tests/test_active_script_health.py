import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RESEARCH_SCRIPTS = (
    Path("scripts/assess_v7_readiness.py"),
    Path("scripts/build_frame_actions_v82.py"),
    Path("scripts/build_frame_tracking_v8.py"),
    Path("scripts/build_temporal_providers_v89.py"),
    Path("scripts/collect_world_model_traces.py"),
    Path("scripts/evaluate_large_shadow_v98.py"),
    Path("scripts/fetch_match_baselines_statsbomb.py"),
    Path("scripts/train_controlled_frame_world_v81.py"),
    Path("scripts/train_frame_world_v8.py"),
    Path("scripts/train_probabilistic_world_v7.py"),
    Path("scripts/train_semantic_frame_world_v82.py"),
    Path("scripts/train_semantic_frame_world_v83.py"),
)


def test_active_research_scripts_are_valid_python():
    for relative_path in ACTIVE_RESEARCH_SCRIPTS:
        source_path = ROOT / relative_path
        ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))


def test_active_research_scripts_do_not_assign_lambdas():
    assignment_types = (ast.Assign, ast.AnnAssign, ast.NamedExpr)
    for relative_path in ACTIVE_RESEARCH_SCRIPTS:
        source_path = ROOT / relative_path
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, assignment_types) and isinstance(node.value, ast.Lambda)
        ]
        assert not offenders, f"{relative_path}: assigned lambdas at {offenders}"
