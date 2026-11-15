from pathlib import Path

from gfs_core.bootstrap.builtin_plugins import build_micro_registry
from gfs_core.kernel import PluginKernel
from gfs_studio.plugins.product.studio.workflow import ProductWorkflows
from gfs_studio.plugins.world_model.trace_codec import WorldModelTraceCodec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = PROJECT_ROOT.parent / "gfs-core"


class RecordingAssessor:
    def __init__(self):
        self.calls = []

    def assess_case(self, case, observation, action):
        self.calls.append(case.case_id)
        return {
            **case.as_dict(),
            "candidate_id": "model_assessment",
            "observed_simulation_metrics": None,
        }


def test_model_assessment_uses_initial_state_and_same_case_identity(tmp_path):
    kernel = PluginKernel(build_micro_registry(CORE_ROOT))
    try:
        kernel.start(["gfs.core.match.workflow"])
        assessor = RecordingAssessor()
        workflows = ProductWorkflows(
            tmp_path,
            kernel.context.require("match-workflow"),
            policy_assessor=assessor,
            trace_codec=WorldModelTraceCodec(),
        )
        result = workflows.compare_tactics("Brazil", "Argentina", 42, seeds=[42])
        assessments = result["evaluation"]["model_assessments"]
        assert len(assessments) == 2
        assert len(set(assessor.calls)) == 1
        assert assessments[0]["case_id"] == result["evaluation"]["model_assessments"][1]["case_id"]
        assert assessments[0]["observed_simulation_metrics"] is None
        assert result["baseline"].goals_micro_home == 0
        assert result["baseline"].goals_micro_away == 3
    finally:
        kernel.stop()
