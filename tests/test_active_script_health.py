import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RESEARCH_SCRIPTS = (
    Path('scripts/assess_v7_readiness.py'),
    Path('scripts/build_frame_actions_v82.py'),
    Path('scripts/build_frame_tracking_v8.py'),
    Path('scripts/build_temporal_providers_v89.py'),
    Path('scripts/collect_world_model_traces.py'),
    Path('scripts/evaluate_large_shadow_v98.py'),
    Path('scripts/fetch_match_baselines_statsbomb.py'),
    Path('scripts/train_frame_world_v8.py'),
    Path('scripts/train_probabilistic_world_v7.py'),
)


def test_active_research_scripts_are_valid_python():
    for relative_path in ACTIVE_RESEARCH_SCRIPTS:
        source_path = ROOT / relative_path
        ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
