from gfs_core.contracts import PluginContext, PluginMetadata

from .current import current_simulation_verification, current_verification

PROTOCOL = "m2_mirrored_policy_protocol_v1.json"
PROGRESS = "m2_mirrored_policy_progress_v6.json"
REPRODUCTION = "reproduction_release_verification_v1.json"


class ReleaseGates:
    def __init__(self, evidence, identities, experiments, participants) -> None:
        self._evidence = evidence
        self._identities = identities
        self._experiments = experiments
        self._participants = participants

    def evaluate(self) -> dict:
        progress = self._evidence.read_frozen(PROGRESS)
        reproduction = self._evidence.read_frozen(REPRODUCTION)
        study = self._experiments.status("v6")
        current = current_verification(self._evidence.root)
        simulation = current_simulation_verification(self._evidence.root)
        participants = self._participants.evaluate()
        protocol_identity = self._identities.register(
            PROTOCOL, progress["execution_identity"]["protocol_sha256"],
        )
        readiness = reproduction.get("readiness", {})
        checks = {
            "identity_integrity": protocol_identity["current"],
            "code_ready": current["code_ready"],
            "current_simulation_evidence": simulation["available"],
            "deployment_runtime": current["runtime_verified"],
            "reproduction_passed": bool(reproduction.get("passed", False)),
            "m2_completed": study.complete,
            "m2_promotion_supported": study.promotion_supported,
            "independent_reproduction": bool(
                readiness.get("independent_reproduction_available", False)
            ),
            "participant_evidence": bool(participants["user_value_validated"]),
        }
        return {
            "release_ready": all(checks.values()),
            "code_ready": checks["code_ready"],
            "status": "ready" if all(checks.values()) else "blocked",
            "checks": checks,
            "open_gates": sorted(name for name, passed in checks.items() if not passed),
            "identities": {PROTOCOL: protocol_identity},
            "current_source_identity": current["source_identity"],
            "current_simulation": {
                "available": simulation["available"],
                "source_identity": simulation["source_identity"],
                "matches_executed": (
                    simulation["report"] or {}
                ).get("matches_executed", 0),
            },
            "participants": {
                key: participants[key]
                for key in (
                    "registered_participants", "valid_participants",
                    "study_complete", "user_value_validated",
                )
            },
            "scope": "current_build_and_historical_research_evidence",
        }


class ReleasePlugin:
    metadata = PluginMetadata(
        id="gfs.studio.evidence.release",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.evidence",
        provides=("release-gates",),
        requires=(
            "gfs.studio.evidence.registry",
            "gfs.studio.research.experiments",
            "gfs.studio.participants.scoring",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("release-gates", ReleaseGates(
            context.require("receipts"),
            context.require("identity-registry"),
            context.require("experiments"),
            context.require("participant-scoring"),
        ))

    def deactivate(self) -> None:
        pass
