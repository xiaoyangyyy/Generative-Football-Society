"""Deterministic sub-tick reception events reconciled at macro boundaries."""

from __future__ import annotations
from dataclasses import dataclass, field
import heapq
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from src.match_engine.continuous_micro_clock import ContinuousMicroEventClock
from src.match_engine.frame_world.calibration import load_calibrated_temporal_router
from src.match_engine.frame_world.pass_triplet import load_pass_triplet
from src.match_engine.frame_world.router import ActionRequest, FrameContext

if TYPE_CHECKING:
    from src.match_engine.passing_engine import PassAction
    from src.match_engine.state import MatchAffectiveState


@dataclass(order=True)
class _QueuedReception:
    due_at: float
    sequence: int
    receiver_id: str = field(compare=False)
    defender_id: str = field(compare=False)
    positions: dict[str, np.ndarray] = field(compare=False)
    velocities: dict[str, np.ndarray] = field(compare=False)
    kind: str = field(compare=False)


class SubtickReceptionQueue:
    """Runs local learned reception responses without changing macro action cadence."""

    def __init__(
        self,
        calibration_artifact: str | Path,
        pass_artifact: str | Path,
        provider: str = "skillcorner",
        blend: float = 0.5,
    ):
        pass_model = load_pass_triplet(pass_artifact)[0].eval()
        self.router = load_calibrated_temporal_router(
            calibration_artifact, pass_models={provider: pass_model}
        )
        if (
            provider not in self.router.temporal_models
            or provider not in self.router.pass_models
        ):
            raise ValueError(f"unsupported sub-tick provider: {provider}")
        if not 0.0 < float(blend) <= 1.0:
            raise ValueError("sub-tick blend must be in (0, 1]")
        self.provider = provider
        self.blend = float(blend)
        self._adapter = ContinuousMicroEventClock.__new__(ContinuousMicroEventClock)
        self._adapter._history = __import__("collections").deque(maxlen=5)
        self._pending: list[_QueuedReception] = []
        self._sequence = 0
        self._pre_context: FrameContext | None = None
        self._pre_index: dict[str, int] = {}
        self.scheduled = 0
        self.applied = 0
        self.cancelled = 0
        self.cancelled_possession = 0
        self.cancelled_missing = 0
        self.cancelled_off_pitch = 0
        self.mismatch_examples: list[tuple[object, ...]] = []
        self.skipped_possessor_mismatch = 0
        self.displacement_sum_m = 0.0
        self.delay_min_s = float("inf")
        self.delay_max_s = 0.0
        self.delay_sum_s = 0.0
        self.kind_counts: dict[str, int] = {}

    def capture_pre_action(self, state: MatchAffectiveState) -> None:
        self._pre_context, self._pre_index = ContinuousMicroEventClock.observe(
            self._adapter, state
        )

    def schedule_after_pass(
        self, state: MatchAffectiveState, action: PassAction | None
    ) -> None:
        context, index = self._pre_context, self._pre_index
        if context is None or action is None or not action.completed:
            return
        if state.ball.possessor_id != action.to_id:
            self.skipped_possessor_mismatch += 1
            return
        actor = index.get(action.from_id, -1)
        receiver = index.get(action.to_id, -1)
        if actor < 0 or receiver < 0:
            return
        opponents = np.flatnonzero(
            context.visible[-1]
            & (context.teams != context.teams[receiver])
            & (context.teams >= 0)
        )
        if not len(opponents):
            return
        defender = int(
            opponents[
                np.argmin(
                    np.linalg.norm(
                        context.positions[-1, opponents]
                        - context.positions[-1, receiver],
                        axis=1,
                    )
                )
            ]
        )
        request = ActionRequest("pass", self.provider, actor, receiver, defender)
        plan = self.router.plan_next_mark(context, request)
        transition = self.router.transition(context, [request])
        if (
            plan is None
            or not transition.decisions
            or not transition.decisions[0].enabled
        ):
            return
        players = {p.player_id: p for p in state.home.players + state.away.players}
        defender_id = next(
            (pid for pid, slot in index.items() if slot == defender), None
        )
        if defender_id is None or action.to_id not in players:
            return
        baseline = np.clip(
            context.positions[-1] + context.velocities[-1] * self.router.horizon_s,
            -0.15,
            1.15,
        )
        positions = {
            action.to_id: (transition.positions[receiver] - baseline[receiver]).copy(),
            defender_id: (transition.positions[defender] - baseline[defender]).copy(),
        }
        velocities = {
            action.to_id: (
                transition.velocities[receiver] - context.velocities[-1, receiver]
            ).copy(),
            defender_id: (
                transition.velocities[defender] - context.velocities[-1, defender]
            ).copy(),
        }
        delay = float(plan.delay_seconds)
        self.delay_min_s = min(self.delay_min_s, delay)
        self.delay_max_s = max(self.delay_max_s, delay)
        self.delay_sum_s += delay
        event = _QueuedReception(
            float(state.clock_seconds) + delay,
            self._sequence,
            action.to_id,
            defender_id,
            positions,
            velocities,
            plan.kind,
        )
        self._sequence += 1
        heapq.heappush(self._pending, event)
        self.scheduled += 1
        self.kind_counts[plan.kind] = self.kind_counts.get(plan.kind, 0) + 1

    def reconcile(self, until: float, state: MatchAffectiveState) -> None:
        players = {p.player_id: p for p in state.home.players + state.away.players}
        while self._pending and self._pending[0].due_at <= float(until):
            event = heapq.heappop(self._pending)
            if state.ball.possessor_id != event.receiver_id:
                self.cancelled += 1
                self.cancelled_possession += 1
                if len(self.mismatch_examples) < 5:
                    self.mismatch_examples.append(
                        (
                            event.receiver_id,
                            state.ball.possessor_id,
                            event.due_at,
                            float(until),
                        )
                    )
                continue
            if event.receiver_id not in players or event.defender_id not in players:
                self.cancelled += 1
                self.cancelled_missing += 1
                continue
            for player_id in (event.receiver_id, event.defender_id):
                player = players[player_id]
                if not player.on_pitch:
                    self.cancelled += 1
                    self.cancelled_off_pitch += 1
                    break
                target = player.position + self.blend * event.positions[player_id]
                delta = (target - player.position) * np.array([105.0, 68.0])
                self.displacement_sum_m += float(np.linalg.norm(delta))
                player.position = np.clip(target, -0.15, 1.15)
                player.velocity = (
                    player.velocity + self.blend * event.velocities[player_id]
                )
            else:
                state.ball.position = players[event.receiver_id].position.copy()
                state.ball.velocity = players[event.receiver_id].velocity.copy()
                self.applied += 1

    def diagnostics(self) -> dict[str, object]:
        return {
            "enabled": True,
            "provider": self.provider,
            "scheduled": self.scheduled,
            "applied": self.applied,
            "cancelled": self.cancelled,
            "cancelled_possession": self.cancelled_possession,
            "cancelled_missing": self.cancelled_missing,
            "cancelled_off_pitch": self.cancelled_off_pitch,
            "mismatch_examples": list(self.mismatch_examples),
            "skipped_possessor_mismatch": self.skipped_possessor_mismatch,
            "pending": len(self._pending),
            "mean_applied_displacement_m": self.displacement_sum_m
            / max(1, 2 * self.applied),
            "delay_s": {
                "min": 0.0 if self.scheduled == 0 else self.delay_min_s,
                "mean": self.delay_sum_s / max(1, self.scheduled),
                "max": self.delay_max_s,
            },
            "next_mark_counts": dict(self.kind_counts),
            "rng_draws": 0,
        }
