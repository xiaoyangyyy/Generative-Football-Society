import shutil
from pathlib import Path

from gfs_core.bootstrap.builtin_plugins import build_micro_registry
from gfs_core.kernel import PluginKernel
from gfs_core.kernel.project import data_identity
from gfs_studio.plugins.evidence.receipts.store import EvidenceStore
from gfs_studio.plugins.evidence.release.current import current_simulation_verification
from gfs_studio.plugins.research.calibration.study import CalibrationService
from gfs_studio.plugins.training.evaluation.validator import CandidateEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = PROJECT_ROOT.parent / "gfs-core"


def test_current_simulation_executes_real_core_and_preserves_scope(tmp_path):
    frozen = tmp_path / "evidence" / "frozen"
    frozen.mkdir(parents=True)
    shutil.copy(
        PROJECT_ROOT / "evidence" / "frozen" / "m0_formal_baseline_v1.json",
        frozen / "m0_formal_baseline_v1.json",
    )
    core = PluginKernel(build_micro_registry(CORE_ROOT))
    try:
        core.start(["gfs.core.match.workflow"])
        service = CalibrationService(
            EvidenceStore(tmp_path),
            CandidateEvaluator,
            core.context.require("match-workflow"),
        )
        report = service.evaluate_current(
            fixtures=["Brazil_vs_Argentina"], sample_indices=[0], fast=True,
        )
        assert report["state"] == "completed"
        assert report["matches_executed"] == 1
        assert report["protocol_complete"] is False
        assert report["protocol"]["engine"] == "micro_fast_demo"
        assert report["metrics"]["pass_completion"]["samples"] == 1
        assert report["rows"][0]["home"] == "Brazil"
        assert report["data_identity"] == data_identity(tmp_path)
        assert report["scope"] == "current_descriptive_external_metric_comparison_not_preregistered_confirmatory"
        assert Path(report["receipt"]["path"]).is_file()
        status = current_simulation_verification(tmp_path)
        assert status["available"] is False
        assert status["report"]["matches_executed"] == 1
    finally:
        core.stop()
