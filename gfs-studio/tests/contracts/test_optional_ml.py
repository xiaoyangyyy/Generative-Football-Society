import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_release_and_web_surfaces_do_not_require_torch(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(PROJECT_ROOT / "src"),
        str(PROJECT_ROOT.parent / "gfs-core" / "src"),
    ])
    code = """
import sys
sys.modules['torch'] = None
from gfs_studio.bootstrap.application import StudioApplication
application = StudioApplication(r'%s')
status = application.release_status()
assert status['release_ready'] is False
application.web_gateway().close()
from gfs_studio.plugins.world_model.model import build_model
try:
    build_model()
except RuntimeError as error:
    assert 'gfs-studio[ml]' in str(error)
else:
    raise AssertionError('ML extra was not required')
print('optional-ml-ok')
""" % PROJECT_ROOT
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "optional-ml-ok" in completed.stdout
