#!/usr/bin/env python3
"""Register, seal, deploy v7 models, and retain an explicit v6 rollback pointer."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data_engine.dataset_registry import file_sha256, write_json_atomic
from src.model_registry import ModelRegistry
def main():
    manifest=ROOT/"data/releases/v7.0.0.json"; release=json.loads(manifest.read_text(encoding="utf-8"))
    if release.get("status")!="frozen" or not all(release["gates"].values()): raise SystemExit("v7 frozen release is not promotable")
    registry=ModelRegistry(ROOT/"data/releases/model_registry.json")
    models={"joint_pass":ROOT/"data/world_model/joint_pass_v7_candidate.json","shot":ROOT/"data/world_model/shot_v7_candidate.json","probabilistic_world":ROOT/"data/world_model/probabilistic_v7_candidate.json"}
    evidence={"release_manifest":"data/releases/v7.0.0.json","release_sha256":file_sha256(manifest),"gates":release["gates"]}
    for name,artifact in models.items():
        key=f"{name}:7.0.0"
        if key not in registry.data["models"]: registry.register(name,"7.0.0",artifact)
        status=registry.data["models"][key]["status"]
        order=("candidate","evaluated","sealed","deployed")
        for target in order[order.index(status)+1:]:
            registry.promote(name,"7.0.0",target,evidence if target in {"sealed","deployed"} else None)
        registry.attest(name,"7.0.0",evidence)
    pointer={"schema_version":1,"active":"7.0.0","manifest":"data/releases/v7.0.0.json","manifest_sha256":file_sha256(manifest),"rollback":{"release":"6.0.0","manifest":"data/releases/v6.0.0.json","manifest_sha256":file_sha256(ROOT/"data/releases/v6.0.0.json")}}
    write_json_atomic(ROOT/"data/releases/current.json",pointer); print(json.dumps(pointer,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
