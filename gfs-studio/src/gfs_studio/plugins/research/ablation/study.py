from gfs_core.contracts import PluginContext, PluginMetadata


class AblationStudy:
    def __init__(self, evidence) -> None:
        self._evidence = evidence

    def summary(self) -> dict:
        document = self._evidence.read_frozen("formal_ablation_summary_v1.json")
        return {
            "protocol": document["protocol"],
            "complete": bool(document["complete"]),
            "variants": document["variants"],
            "evidence": "frozen_historical",
            "new_runs": 0,
            "causal_claim": False,
        }

    def variant(self, name: str) -> dict:
        variants = self._evidence.read_frozen(
            "formal_ablation_summary_v1.json"
        )["variants"]
        if name not in variants:
            raise KeyError(f"unknown frozen ablation variant: {name}")
        return {
            "name": name,
            "result": variants[name],
            "evidence": "frozen_historical",
            "new_runs": 0,
        }


class AblationPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.research.ablation",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.research",
        provides=("ablation",),
        requires=("gfs.studio.evidence.receipts",),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("ablation", AblationStudy(context.require("receipts")))

    def deactivate(self) -> None:
        pass
