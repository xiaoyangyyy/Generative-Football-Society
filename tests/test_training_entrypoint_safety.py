import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ensure_world_model_force_requires_explicit_training_authority():
    result = subprocess.run(
        [sys.executable, "scripts/ensure_world_model.py", "--force"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "requires explicit --train authorization" in result.stderr


def test_world_model_dataset_identity_changes_with_content():
    import numpy as np
    from scripts.train_world_model import _dataset_identity

    first = np.asarray([[1.0, 2.0]], dtype=np.float32)
    second = np.asarray([[1.0, 3.0]], dtype=np.float32)
    assert _dataset_identity(first) == _dataset_identity(first.copy())
    assert _dataset_identity(first) != _dataset_identity(second)
