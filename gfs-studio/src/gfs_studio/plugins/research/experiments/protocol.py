from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_studio.contracts.research import M2StudyStatus

PROTOCOL = "m2_mirrored_policy_protocol_v1.json"
V1_PROGRESS = "m2_mirrored_policy_progress_v1.json"
V6_PROGRESS = "m2_mirrored_policy_progress_v6.json"


class M2ResearchService:
    def __init__(self, evidence) -> None:
        self.evidence = evidence

    def status(self, version: str = "v6") -> M2StudyStatus:
        if version == "v6":
            progress = self.evidence.read_frozen(V6_PROGRESS)
            identity_status = (
                "current" if self.evidence.verify_current_identity(
                    PROTOCOL,
                    progress["execution_identity"]["protocol_sha256"],
                ) else "identity_mismatch"
            )
            next_action = (
                "analyze_frozen_results" if progress["state"] == "completed"
                else "resume_identity_matched_m2_study"
            )
            analysis = progress.get("analysis") or {}
            return M2StudyStatus(
                progress["execution_identity"].get("protocol_path", "m2-mirrored-policy-v1").split("/")[-1],
                "v6",
                progress["state"],
                _runs(progress),
                _budget(self.evidence),
                bool(analysis.get("promotion_supported", False)),
                next_action,
                identity_status,
            )
        if version == "v1":
            progress = self.evidence.read_frozen(V1_PROGRESS)
            analysis = progress.get("analysis") or {}
            return M2StudyStatus(
                "m2-mirrored-policy-v1",
                "v1",
                progress["state"],
                _runs(progress),
                _budget(self.evidence),
                bool(analysis.get("promotion_supported", False)),
                str(analysis.get("decision", "retain_research_only_and_diagnose")),
                "frozen_historical",
            )
        raise ValueError(f"unsupported M2 study version: {version}")

    def history(self) -> dict:
        return {
            "active": self.status("v6").as_dict(),
            "previous": self.status("v1").as_dict(),
        }

    def release_status(self) -> dict:
        reproduction = self.evidence.read_frozen(
            "reproduction_release_verification_v1.json"
        )
        excellence = self.evidence.read_frozen(
            "excellence_score_verification_v1.json"
        )
        checks = reproduction.get("checks", {})
        return {
            "release_ready": False,
            "code_ready": bool(reproduction.get("code_ready", False)),
            "status": reproduction.get("status", "failed"),
            "open_gates": sorted(
                key for key, passed in checks.items() if not passed
            ),
            "scores": excellence.get("scores", {}),
            "next_action": "repair_current_evidence_and_reproduction_gates",
        }


class ExperimentsPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.research.experiments",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.research",
        provides=("experiments",),
        requires=("gfs.studio.evidence.receipts",),
    )

    def __init__(self, evidence) -> None:
        self._evidence = evidence

    def activate(self, context: PluginContext) -> None:
        context.provide("experiments", M2ResearchService(self._evidence))

    def deactivate(self) -> None:
        pass


def _runs(progress) -> int:
    return sum(len(arm.get("rows", [])) for arm in progress.get("arms", {}).values())


def _budget(evidence) -> int:
    protocol = evidence.read_frozen(PROTOCOL)
    return int(protocol["design"]["runs_total"])
