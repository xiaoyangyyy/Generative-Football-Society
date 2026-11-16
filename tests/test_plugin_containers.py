import json
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


def test_studio_uses_local_core_without_pulling_its_private_tag(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "1" * 40)
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "docker")
    builds = []

    def fake_run(command, timeout=60):
        del timeout
        if command[1] == "build":
            builds.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1:3] == ["image", "inspect"]:
            image = command[3]
            payload = [{
                "Id": "sha256:" + ("1" if "core:" in image else "2") * 64,
                "Config": {
                    "User": "10001:10001",
                    "Labels": {"org.opencontainers.image.revision": "1" * 40},
                },
            }]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if command[1] == "run":
            return subprocess.CompletedProcess(command, 1, "", "stop after build")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime, "_run", fake_run)
    report = runtime.verify(execute=True)
    assert report["status"] == "core_cli_failed"
    assert len(builds) == 2
    assert "--pull" in builds[0]
    assert "--pull" not in builds[1]
    assert f"CORE_IMAGE={runtime.CORE_IMAGE}-verification-" in " ".join(builds[1])
