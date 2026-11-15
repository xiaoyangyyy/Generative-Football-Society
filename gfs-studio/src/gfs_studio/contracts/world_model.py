from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

OBS_DIM = 307
ACTION_DIM = 18


@runtime_checkable
class TraceCodec(Protocol):
    version: str

    def observation(self, state: object) -> np.ndarray: ...

    def action(self, kind: str, target, horizon_s: float) -> np.ndarray: ...

    def strip_action(self, action) -> np.ndarray: ...


@runtime_checkable
class CheckpointCodec(Protocol):
    def load(self, path: str) -> tuple[object, object, dict]: ...

    def save(self, path: str, model: object, config: object, metadata: dict) -> None: ...


@runtime_checkable
class LatentModels(Protocol):
    def load(self, path: str) -> object: ...


@runtime_checkable
class Planner(Protocol):
    def forecast(self, observation, action, **options) -> dict: ...


@runtime_checkable
class FrameCatalog(Protocol):
    def manifest(self) -> dict: ...
