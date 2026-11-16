from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from gfs_core.kernel import PluginKernel
from gfs_studio.bootstrap.builtin_plugins import (
    build_cognition_registry,
    build_recovery_registry,
    build_planning_registry,
    build_research_registry,
    build_task_registry,
    build_web_registry,
    build_training_registry,
    build_world_model_registry,
)
from gfs_studio.plugins.training.datasets.manifest import TraceDataset

_EXPERIMENTS_PLUGIN = "gfs.studio.research.experiments"
_LATENT_PLUGIN = "gfs.studio.world-model.latent"
_TRAINERS_PLUGIN = "gfs.studio.training.trainers"
_COGNITION_PLUGIN = "gfs.studio.cognition.gateway"
_WEB_PLUGIN = "gfs.studio.web"
_RECOVERY_PLUGIN = "gfs.studio.product.recovery"
_TASKS_PLUGIN = "gfs.studio.product.tasks"
_PLANNING_PLUGIN = "gfs.studio.world-model.planning"
_POLICIES_PLUGIN = "gfs.studio.world-model.policies"
_FRAME_PLUGIN = "gfs.studio.world-model.frame"


class StudioApplication:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._cognition_registry = build_cognition_registry(self.project_root)
        self._web_kernel = None

    def m2_status(self) -> dict:
        kernel = PluginKernel(build_research_registry(self.project_root))
        try:
            kernel.start([_EXPERIMENTS_PLUGIN])
            return kernel.context.require("experiments").history()
        finally:
            kernel.stop()

    def evaluate_current_simulation(self, *, fixtures=None, sample_indices=None, fast: bool = False) -> dict:
        kernel = PluginKernel(build_research_registry(self.project_root))
        try:
            kernel.start(["gfs.studio.research.calibration"])
            return kernel.context.require("calibration").evaluate_current(
                fixtures=fixtures, sample_indices=sample_indices, fast=fast,
            )
        finally:
            kernel.stop()

    def release_status(self) -> dict:
        kernel = PluginKernel(build_research_registry(self.project_root))
        try:
            kernel.start(["gfs.studio.evidence.release"])
            historical = kernel.context.require("experiments").release_status()
            gates = kernel.context.require("release-gates").evaluate()
            return {
                **historical,
                "code_ready": gates["code_ready"],
                "checks": gates["checks"],
                "identities": gates["identities"],
                "current_source_identity": gates["current_source_identity"],
                "current_simulation": gates["current_simulation"],
                "participants": gates["participants"],
                "open_gates": sorted(set(historical["open_gates"] + gates["open_gates"])),
                "scope": gates["scope"],
            }
        finally:
            kernel.stop()

    def world_model_status(self, checkpoint: str) -> dict:
        runtime = self._load_world_model(checkpoint)
        return runtime.status()

    def evaluate_rollout(self, trace, checkpoint: str, *, horizons=(1, 2, 4)) -> dict:
        kernel = PluginKernel(build_training_registry(self.project_root))
        try:
            kernel.start(["gfs.studio.training.evaluation"])
            dataset = TraceDataset.from_jsonl(trace)
            runtime = kernel.context.require("latent-models").load(
                self.project_root / "data" / "world-model" / checkpoint
            )
            return kernel.context.require("model-evaluation").evaluate_rollout(
                runtime.model,
                dataset,
                kernel.context.require("trace-codec"),
                horizons=horizons,
            )
        finally:
            kernel.stop()

    def imagine(self, observation, action, *, checkpoint: str, steps: int | None = None):
        runtime = self._load_world_model(checkpoint)
        return runtime.imagine(observation, action, steps=steps)

    def _load_world_model(self, checkpoint: str):
        kernel = PluginKernel(build_world_model_registry(self.project_root))
        try:
            kernel.start([_LATENT_PLUGIN])
            runtime_type = kernel.context.require("latent-models")
            return runtime_type.load(
                self.project_root / "data" / "world-model" / checkpoint
            )
        finally:
            kernel.stop()

    def train(self, trace, run_name: str, *, epochs: int = 1, resume: bool = False, run_seed: int = 42) -> dict:
        from gfs_studio.plugins.training.trainers.determinism import reproducible
        from gfs_studio.plugins.world_model.model import build_model

        kernel = PluginKernel(build_training_registry(self.project_root))
        try:
            kernel.start([_TRAINERS_PLUGIN])
            dataset = TraceDataset.from_jsonl(trace)
            with reproducible(run_seed):
                model = build_model()
                result = kernel.context.require("trainers").run(
                    model,
                    dataset,
                    run_name,
                    epochs=epochs,
                    resume=resume,
                    run_seed=run_seed,
                )
            return {
                "status": result.status,
                "epochs": result.epochs,
                "checkpoint": result.checkpoint,
                "validation": result.validation,
                "qualification": result.qualification,
                "resumed": result.resumed,
                "dataset_identity": result.dataset_identity,
                "code_identity": result.code_identity,
            }
        finally:
            kernel.stop()

    def export_traces(self, output, *, home: str = "Brazil", away: str = "Argentina", seed: int = 42, fast: bool = True, stop_at_tick: int | None = None) -> dict:
        kernel = PluginKernel(build_training_registry(self.project_root))
        try:
            kernel.start(["gfs.studio.training.trace-export"])
            return kernel.context.require("trace-export").export(
                output,
                home=home,
                away=away,
                seed=seed,
                fast=fast,
                stop_at_tick=stop_at_tick,
            )
        finally:
            kernel.stop()

    def export_cohort(self, specifications: list[dict], output_dir, *, fast: bool = True, stop_at_tick: int | None = None, holdout_teams: list[str] | None = None, validation_time_fraction: float = 0.2) -> dict:
        kernel = PluginKernel(build_training_registry(self.project_root))
        try:
            kernel.start(["gfs.studio.training.trace-export"])
            return kernel.context.require("trace-export").export_cohort(
                specifications,
                output_dir,
                fast=fast,
                stop_at_tick=stop_at_tick,
                holdout_teams=holdout_teams,
                validation_time_fraction=validation_time_fraction,
            )
        finally:
            kernel.stop()

    def overview(self) -> dict:
        return {
            "m2": self.m2_status(),
            "release": self.release_status(),
            "cognition": self.cognition_status(),
        }

    def cognition_status(self) -> dict:
        kernel = PluginKernel(self._cognition_registry)
        try:
            kernel.start([_COGNITION_PLUGIN])
            return kernel.context.require("provider-gateway").status()
        finally:
            kernel.stop()

    def reflect(self, *, role: str, topic: str, match_id: str, seed: int = 42):
        kernel = PluginKernel(self._cognition_registry)
        try:
            kernel.start(["gfs.studio.cognition.reflection"])
            return kernel.context.require("reflection").reflect(
                role=role, topic=topic, match_id=match_id, seed=seed,
            )
        finally:
            kernel.stop()

    def web_gateway(self):
        if self._web_kernel is None:
            self._web_kernel = PluginKernel(build_web_registry(self.project_root))
            self._web_kernel.start([_WEB_PLUGIN])
        return self._web_kernel.context.require("http-gateway")

    def close(self) -> None:
        if self._web_kernel is not None:
            self._web_kernel.stop()
            self._web_kernel = None

    def backup_create(self) -> dict:
        kernel = PluginKernel(build_recovery_registry(self.project_root))
        try:
            kernel.start([_RECOVERY_PLUGIN])
            return kernel.context.require("product-recovery").create()
        finally:
            kernel.stop()

    def backup_verify(self, backup_id: str) -> dict:
        kernel = PluginKernel(build_recovery_registry(self.project_root))
        try:
            kernel.start([_RECOVERY_PLUGIN])
            return kernel.context.require("product-recovery").verify(backup_id)
        finally:
            kernel.stop()

    def backup_restore(self, backup_id: str, destination, *, replace: bool = False) -> dict:
        kernel = PluginKernel(build_recovery_registry(self.project_root))
        try:
            kernel.start([_RECOVERY_PLUGIN])
            return kernel.context.require("product-recovery").restore(
                backup_id, destination, replace=replace,
            )
        finally:
            kernel.stop()

    def submit_task(self, home: str, away: str, seed: int, idempotency_key: str) -> dict:
        kernel = PluginKernel(build_task_registry(self.project_root))
        try:
            kernel.start([_TASKS_PLUGIN])
            return kernel.context.require("product-tasks").submit(
                idempotency_key,
                {"home": home, "away": away, "seed": seed},
            )
        finally:
            kernel.stop()

    def task_get(self, task_id: str) -> dict | None:
        kernel = PluginKernel(build_task_registry(self.project_root))
        try:
            kernel.start([_TASKS_PLUGIN])
            return kernel.context.require("product-tasks").get(task_id)
        finally:
            kernel.stop()

    def task_list(self) -> list[dict]:
        kernel = PluginKernel(build_task_registry(self.project_root))
        try:
            kernel.start([_TASKS_PLUGIN])
            return kernel.context.require("product-tasks").list()
        finally:
            kernel.stop()

    def task_wait(self, task_id: str, timeout: float = 30.0) -> dict:
        kernel = PluginKernel(build_task_registry(self.project_root))
        try:
            kernel.start([_TASKS_PLUGIN])
            return kernel.context.require("product-tasks").wait(task_id, timeout)
        finally:
            kernel.stop()

    def task_recover(self) -> int:
        kernel = PluginKernel(build_task_registry(self.project_root))
        try:
            kernel.start([_TASKS_PLUGIN])
            return kernel.context.require("product-tasks").recover()
        finally:
            kernel.stop()

    def task_resume(self, task_id: str | None = None) -> int:
        kernel = PluginKernel(build_task_registry(self.project_root))
        try:
            kernel.start([_TASKS_PLUGIN])
            return kernel.context.require("product-tasks").resume(task_id)
        finally:
            kernel.stop()

    def plan(self, observation, action, *, steps: int | None = None, kind: str = "general") -> dict:
        kernel = PluginKernel(build_planning_registry(self.project_root))
        try:
            kernel.start([_PLANNING_PLUGIN])
            return kernel.context.require("planning").forecast(
                observation, action, steps=steps, kind=kind,
            )
        finally:
            kernel.stop()

    def recommend(self, observation, action, *, baseline_action: str = "pass") -> dict:
        kernel = PluginKernel(build_planning_registry(self.project_root))
        try:
            kernel.start([_POLICIES_PLUGIN])
            return kernel.context.require("policies").recommend(
                observation, action, baseline_action=baseline_action,
            )
        finally:
            kernel.stop()

    def frame_manifest(self) -> dict:
        kernel = PluginKernel(build_planning_registry(self.project_root))
        try:
            kernel.start([_FRAME_PLUGIN])
            return kernel.context.require("frame-models").manifest()
        finally:
            kernel.stop()

    def run_cli(self, arguments: Sequence[str]) -> str:
        if list(arguments) == ["frame", "status"]:
            return json.dumps(self.frame_manifest(), ensure_ascii=True, indent=2)
        if list(arguments) == ["tasks"]:
            return json.dumps(self.task_list(), ensure_ascii=True, indent=2)
        if list(arguments) == ["tasks", "recover"]:
            return json.dumps({"recovered": self.task_recover()}, ensure_ascii=True, indent=2)
        if list(arguments) == ["tasks", "resume"]:
            return json.dumps({"resumed": self.task_resume()}, ensure_ascii=True, indent=2)
        if arguments and arguments[0] == "backup":
            parser = argparse.ArgumentParser(prog="gfs-studio backup")
            parser.add_argument("action", choices=["create", "verify", "restore"])
            parser.add_argument("--backup-id")
            parser.add_argument("--destination")
            parser.add_argument("--replace", action="store_true")
            args = parser.parse_args(list(arguments[1:]))
            if args.action == "create":
                return json.dumps(self.backup_create(), ensure_ascii=True, indent=2)
            if not args.backup_id:
                parser.error("--backup-id is required")
            if args.action == "verify":
                return json.dumps(self.backup_verify(args.backup_id), ensure_ascii=True, indent=2)
            if not args.destination:
                parser.error("--destination is required")
            return json.dumps(
                self.backup_restore(args.backup_id, args.destination, replace=args.replace),
                ensure_ascii=True,
                indent=2,
            )
        if list(arguments) == ["web"]:
            gateway = self.web_gateway()
            server = gateway.serve()
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
                self.close()
            return ""
        if list(arguments) == ["cognition", "status"]:
            return json.dumps(self.cognition_status(), ensure_ascii=True, indent=2)
        if list(arguments[:2]) == ["traces", "export"]:
            parser = argparse.ArgumentParser(prog="gfs-studio traces export")
            parser.add_argument("--output", required=True)
            parser.add_argument("--home", default="Brazil")
            parser.add_argument("--away", default="Argentina")
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--full", action="store_true")
            parser.add_argument("--stop-at-tick", type=int)
            args = parser.parse_args(list(arguments[2:]))
            return json.dumps(self.export_traces(
                args.output,
                home=args.home,
                away=args.away,
                seed=args.seed,
                fast=not args.full,
                stop_at_tick=args.stop_at_tick,
            ), ensure_ascii=True, indent=2)
        if arguments and arguments[0] == "train":
            parser = argparse.ArgumentParser(prog="gfs-studio train")
            parser.add_argument("--trace", required=True)
            parser.add_argument("--run-name", required=True)
            parser.add_argument("--epochs", type=int, default=1)
            parser.add_argument("--resume", action="store_true")
            parser.add_argument("--run-seed", type=int, default=42)
            args = parser.parse_args(list(arguments[1:]))
            return json.dumps(
                self.train(args.trace, args.run_name, epochs=args.epochs, resume=args.resume, run_seed=args.run_seed),
                ensure_ascii=True,
                indent=2,
            )
        if list(arguments) == ["world-model", "status"]:
            return json.dumps(
                self.world_model_status("latent_wm_rollout_calibrated_candidate.pt"),
                ensure_ascii=True,
                indent=2,
            )
        if list(arguments[:2]) == ["research", "evaluate-current"]:
            parser = argparse.ArgumentParser(prog="gfs-studio research evaluate-current")
            parser.add_argument("--fast", action="store_true")
            parser.add_argument("--fixture", action="append")
            parser.add_argument("--sample-index", action="append", type=int)
            args = parser.parse_args(list(arguments[2:]))
            return json.dumps(self.evaluate_current_simulation(
                fixtures=args.fixture,
                sample_indices=args.sample_index,
                fast=args.fast,
            ), ensure_ascii=True, indent=2)
        if list(arguments) == ["research", "status"]:
            return json.dumps(self.m2_status(), ensure_ascii=True, indent=2)
        if list(arguments) == ["release", "status"]:
            return json.dumps(self.release_status(), ensure_ascii=True, indent=2)
        raise ValueError("expected 'research status' or 'release status'")
