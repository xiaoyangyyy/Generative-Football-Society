import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_third_party_plugin_wheel_uses_the_public_entry_point_contract(tmp_path):
    wheel = _build_plugin_wheel(tmp_path)
    installed = tmp_path / "installed"
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-deps", "--target", str(installed), str(wheel),
    ], check=True, capture_output=True, text=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(installed), str(PROJECT_ROOT / "src"),
    ])
    completed = subprocess.run([
        sys.executable, "-c",
        "from gfs_core.kernel import PluginKernel,PluginRegistry,register_entry_points; "
        "registry=PluginRegistry(); "
        "register_entry_points(registry,'gfs_core.plugins'); "
        "kernel=PluginKernel(registry).start(['example.external']); "
        "assert kernel.context.require('external').metadata.id=='example.external'; "
        "kernel.stop(); print('external-wheel-plugin-ok')",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "external-wheel-plugin-ok" in completed.stdout


def _build_plugin_wheel(root: Path) -> Path:
    distribution = "gfs_external_plugin"
    version = "1.0.0"
    dist_info = f"{distribution}-{version}.dist-info"
    documents = {
        "external_plugin.py": (
            "from gfs_core.contracts import PluginMetadata\n"
            "class ExternalPlugin:\n"
            "    metadata=PluginMetadata(id='example.external',version='1.0.0',"
            "api=1,contract='gfs_core.contracts.plugin',provides=('external',))\n"
            "    def activate(self,context): context.provide('external',self)\n"
            "    def deactivate(self): pass\n"
        ),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\nName: gfs-external-plugin\nVersion: 1.0.0\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: gfs-contract-test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": (
            "[gfs_core.plugins]\nexternal = external_plugin:ExternalPlugin\n"
        ),
        f"{dist_info}/RECORD": "",
    }
    wheel = root / f"{distribution}-{version}-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        for name, content in documents.items():
            archive.writestr(name, content)
    return wheel
