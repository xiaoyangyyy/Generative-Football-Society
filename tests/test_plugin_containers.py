import subprocess

from scripts import verify_plugin_containers as runtime


def test_build_failure_has_explicit_status_and_no_leaked_resources(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "1" * 40)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "docker")

    def fail_build(command, timeout=60):
        del timeout
        return subprocess.CompletedProcess(command, 2, "", "build failed")

    monkeypatch.setattr(runtime, "_run", fail_build)
    report = runtime.verify(execute=True)
    assert report["status"] == "core_build_failed"
    assert report["passed"] is False
    assert report["checks"]["ephemeral_container_removed"] is True
    assert report["checks"]["verification_images_removed"] is True
