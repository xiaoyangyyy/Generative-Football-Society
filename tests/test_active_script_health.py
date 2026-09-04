import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DATA_BUILDERS = (
    Path('scripts/build_frame_tracking_v8.py'),
    Path('scripts/fetch_match_baselines_statsbomb.py'),
)


def test_active_data_builders_are_valid_python():
    for relative_path in ACTIVE_DATA_BUILDERS:
        source_path = ROOT / relative_path
        ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
