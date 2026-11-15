from pathlib import Path

from gfs_core.kernel import PluginKernel, PluginRegistry
from gfs_core.kernel.workspace import WorkspaceRepository
from gfs_studio.plugins.cognition.gateway.provider import LLMGateway, RuleBasedProvider
from gfs_studio.plugins.cognition.gateway.service import CognitionPlugin
from gfs_studio.plugins.cognition.reflection.service import ReflectionPlugin
from gfs_studio.plugins.evidence.receipts.store import ReceiptsPlugin


class CountingProvider(RuleBasedProvider):
    def __init__(self):
        self.calls = 0

    def complete(self, request, request_id):
        self.calls += 1
        return super().complete(request, request_id)


def test_reflection_persists_response_then_applies_without_repeated_provider(tmp_path):
    provider = CountingProvider()
    kernel = PluginKernel(PluginRegistry([
        ReceiptsPlugin(tmp_path),
        CognitionPlugin(LLMGateway(provider)),
        ReflectionPlugin(tmp_path),
    ]))
    try:
        kernel.start(["gfs.studio.cognition.reflection"])
        reflection = kernel.context.require("reflection")
        first = reflection.reflect(role="coach", topic="review", match_id="match-1")
        assert provider.calls == 1
        assert first["application"]["official_simulation_changed"] is False
        assert Path(first["receipt"]["path"]).is_file()
        replay = reflection.reflect(role="coach", topic="review", match_id="match-1")
        assert provider.calls == 1
        assert replay["application"] == first["application"]
        record = WorkspaceRepository(tmp_path).read(f"reflections/{first['reflection_id']}.json")
        record["state"] = "response_persisted"
        record["application"] = None
        WorkspaceRepository(tmp_path).write(f"reflections/{first['reflection_id']}.json", record)
        recovered = reflection.resume(first["reflection_id"])
        assert provider.calls == 1
        assert recovered["application"] == first["application"]
        assert reflection.resume(first["reflection_id"])["application"] == first["application"]
    finally:
        kernel.stop()
