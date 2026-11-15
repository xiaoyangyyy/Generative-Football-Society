from gfs_core.contracts import PluginContext, PluginMetadata


class ReproductionService:
    def __init__(self, evidence, identities, experiments) -> None:
        self._evidence = evidence
        self._identities = identities
        self._experiments = experiments

    def status(self) -> dict:
        protocol = self._evidence.read_frozen(
            "independent_reproduction_protocol_verification_v1.json"
        )
        release = self._evidence.read_frozen(
            "reproduction_release_verification_v1.json"
        )
        identity = self._identities.register(
            "independent_reproduction_protocol_v1.json",
            protocol["protocol_sha256"],
        )
        return {
            "status": protocol["status"],
            "ready_to_start": bool(protocol["ready_to_start"]),
            "independent_reproduced": False,
            "release_passed": bool(release.get("passed", False)),
            "protocol_identity": identity,
            "m2": self._experiments.status("v6").as_dict(),
            "evidence": "frozen_historical",
            "new_runs": 0,
            "provider_calls": 0,
        }


class ReproductionPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.research.reproduction",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.research",
        provides=("reproduction",),
        requires=(
            "gfs.studio.research.experiments",
            "gfs.studio.evidence.registry",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("reproduction", ReproductionService(
            context.require("receipts"),
            context.require("identity-registry"),
            context.require("experiments"),
        ))

    def deactivate(self) -> None:
        pass
