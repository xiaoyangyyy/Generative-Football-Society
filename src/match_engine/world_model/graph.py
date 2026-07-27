"""Dynamic interaction graph derived from the stable world-model observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.match_engine.world_model.schema import BALL, PLAYERS


@dataclass(frozen=True)
class InteractionGraph:
    nodes: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


def build_interaction_graph(obs: np.ndarray, *, radius: float = 0.35) -> InteractionGraph:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    players = arr[PLAYERS].reshape(22, 4)
    ball = arr[BALL][:2]
    team = np.concatenate([np.zeros(11), np.ones(11)]).astype(np.float32)
    nodes = np.column_stack([players, team, np.linalg.norm(players[:, :2] - ball, axis=1)])
    senders, receivers, features = [], [], []
    for source in range(22):
        for target in range(22):
            if source == target:
                continue
            delta = players[target, :2] - players[source, :2]
            distance = float(np.linalg.norm(delta))
            if distance > radius:
                continue
            relative_velocity = players[target, 2:4] - players[source, 2:4]
            senders.append(source)
            receivers.append(target)
            features.append([
                delta[0], delta[1], distance,
                relative_velocity[0], relative_velocity[1],
                float(team[source] == team[target]),
            ])
    edge_index = np.asarray([senders, receivers], dtype=np.int64)
    edge_features = np.asarray(features, dtype=np.float32).reshape(-1, 6)
    return InteractionGraph(nodes.astype(np.float32), edge_index, edge_features)
