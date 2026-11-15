from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from gfs_core.kernel.lease import FileLease
from gfs_studio.contracts.training import TrainingResult


class TrainingJob:
    def __init__(self, model, dataset, run_dir: str | Path, checkpoints, evaluator, code_identity: str) -> None:
        self.model = model
        self.dataset = dataset
        self.run_dir = Path(run_dir)
        self.checkpoints = checkpoints
        self.evaluator = evaluator
        self.code_identity = code_identity
        self.train, self.validation, self.split = dataset.split()
        self.configuration = {}

    def run(
        self,
        *,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        resume: bool = False,
        run_seed: int = 42,
    ) -> TrainingResult:
        if (
            isinstance(epochs, bool) or epochs < 1
            or isinstance(batch_size, bool) or batch_size < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or isinstance(run_seed, bool) or not 0 <= run_seed < 2**32
        ):
            raise ValueError("invalid training configuration")
        if not resume and self.run_dir.exists():
            raise FileExistsError(f"training run already exists: {self.run_dir.name}")
        self.configuration = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "run_seed": run_seed,
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with FileLease(self.run_dir / "training.lock"):
            start_epoch, history = self._initialize(resume)
            if epochs < start_epoch:
                raise ValueError("requested epochs precede the checkpoint")
            if start_epoch == 0:
                _atomic_json(self.run_dir / "run.json", self._run_identity())
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate,
            )
            if start_epoch:
                optimizer.load_state_dict(
                    torch.load(
                        self.run_dir / "optimizer.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                torch.set_rng_state(torch.load(
                    self.run_dir / "rng.pt", map_location="cpu", weights_only=True,
                ))
            try:
                self._status("running", start_epoch, history)
                for epoch in range(start_epoch, epochs):
                    loss = self._epoch(optimizer, batch_size, epoch)
                    validation = self.evaluator.evaluate(
                        self.model, self.validation,
                    )
                    history.append({
                        "epoch": epoch + 1,
                        "train_mse": loss,
                        "validation_mse": validation["model_mse"],
                        "persistence_mse": validation["persistence_mse"],
                    })
                    self._save_checkpoint(history, validation)
                    _atomic_torch(optimizer.state_dict(), self.run_dir / "optimizer.pt")
                    _atomic_torch(torch.get_rng_state(), self.run_dir / "rng.pt")
                    self._status("running", epoch + 1, history)
                qualification = self._qualification(history[-1])
                self._save_checkpoint(history, qualification)
                self._status("completed", epochs, history)
                return TrainingResult(
                    "completed",
                    epochs,
                    str(self.run_dir / "latest.pt"),
                    asdict(self.split),
                    qualification,
                    resumed=start_epoch > 0,
                    dataset_identity=self.dataset.identity,
                    code_identity=self.code_identity,
                )
            except BaseException as error:
                current_epoch = history[-1]["epoch"] if history else start_epoch
                self._status("failed", current_epoch, history, error=error)
                raise

    def _initialize(self, resume: bool):
        if not resume:
            return 0, []
        checkpoint = self.run_dir / "latest.pt"
        status_path = self.run_dir / "status.json"
        manifest_path = self.run_dir / "run.json"
        for path in (checkpoint, status_path, manifest_path, self.run_dir / "optimizer.pt", self.run_dir / "rng.pt"):
            if not path.is_file():
                raise FileNotFoundError(f"incomplete training run: {path}")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != self._run_identity():
            raise ValueError("training resume identity mismatch")
        model, _, _ = self.checkpoints.load(str(checkpoint))
        self.model.load_state_dict(model.state_dict())
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return int(status["epoch"]), list(status["history"])

    def _run_identity(self) -> dict:
        return {
            "schema_version": 1,
            "dataset_identity": self.dataset.identity,
            "dataset_source": self.dataset.source,
            "dataset_config_identity": self.dataset.config_identity,
            "code_identity": self.code_identity,
            "configuration": self.configuration,
        }

    def _epoch(self, optimizer, batch_size: int, epoch: int) -> float:
        self.model.train()
        rng = np.random.default_rng(self.configuration["run_seed"] + epoch)
        indices = rng.permutation(len(self.train.observations))
        losses = []
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            observations = torch.from_numpy(self.train.observations[selected])
            actions = torch.from_numpy(self.train.actions[selected])
            targets = torch.from_numpy(self.train.targets[selected])
            optimizer.zero_grad()
            predictions = self.model(observations, actions)[1]
            loss = torch.mean((predictions - targets) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def _save_checkpoint(self, history, qualification) -> None:
        self.checkpoints.save(
            str(self.run_dir / "latest.pt"),
            self.model,
            self.model.cfg,
            {
                "rows": len(self.dataset.observations),
                "groups": len(set(self.dataset.groups.tolist())),
                "validation": qualification,
                "history": history,
                **self._run_identity(),
            },
        )

    def _status(self, state: str, epoch: int, history, *, error=None) -> None:
        _atomic_json(self.run_dir / "status.json", {
            "state": state,
            "epoch": epoch,
            "history": history,
            "error_type": None if error is None else type(error).__name__,
        })

    def _qualification(self, last) -> dict:
        return {
            "groups": self.split.validation_groups,
            "rows": self.split.validation_rows,
            "model_mse": last["validation_mse"],
            "persistence_mse": last["persistence_mse"],
            "eligible": last["validation_mse"] < last["persistence_mse"],
            "scope": "transition_prediction_only_not_policy_or_deployment",
        }


def _atomic_json(path: Path, payload) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode()
    _atomic_bytes(path, encoded)


def _atomic_torch(payload, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _atomic_bytes(path: Path, encoded: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
