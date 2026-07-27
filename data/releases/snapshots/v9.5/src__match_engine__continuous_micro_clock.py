"""Adapter from micro-match state to the calibrated frame-world event clock."""
from __future__ import annotations
from collections import Counter, deque
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from src.match_engine.frame_world.calibration import load_calibrated_temporal_router
from src.match_engine.frame_world.router import ActionRequest, FrameContext
from src.match_engine.frame_world.schema import BALL_INDEX, ENTITY_COUNT
if TYPE_CHECKING:
    from src.match_engine.passing_engine import PassAction
    from src.match_engine.state import MatchAffectiveState, PlayerAffectiveState

class ContinuousMicroEventClock:
    """Schedules post-reception decisions without owning action selection."""
    def __init__(self, artifact: str | Path, provider: str, max_delay_s: float = 5.0):
        self.router = load_calibrated_temporal_router(artifact)
        if provider not in self.router.temporal_models:
            raise ValueError(f"unsupported continuous-clock provider: {provider}")
        self.provider = provider
        self.max_delay_s = float(max_delay_s)
        self.next_action_at = 0.0
        self.pending_kind: str | None = None
        self._history: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=5)
        self.plans = 0
        self.gated_ticks = 0
        self.kind_counts: Counter[str] = Counter()
        self.executed_counts: Counter[str] = Counter()
        self.delay_sum_s = 0.0

    @staticmethod
    def _players(state: MatchAffectiveState) -> list[PlayerAffectiveState]:
        return [*[p for p in state.home.players if p.on_pitch][:16], *[p for p in state.away.players if p.on_pitch][:16]]

    def observe(self, state: MatchAffectiveState) -> tuple[FrameContext, dict[str, int]]:
        positions = np.zeros((ENTITY_COUNT, 2), dtype=np.float32)
        velocities = np.zeros_like(positions)
        visible = np.zeros(ENTITY_COUNT, dtype=bool)
        teams = np.full(ENTITY_COUNT, -1, dtype=np.int64)
        index: dict[str, int] = {}
        for slot, player in enumerate(self._players(state)):
            positions[slot], velocities[slot], visible[slot] = player.position, player.velocity, True
            teams[slot] = 0 if player.team_id == state.home.team_id else 1
            index[player.player_id] = slot
        positions[BALL_INDEX], velocities[BALL_INDEX], visible[BALL_INDEX] = state.ball.position, state.ball.velocity, True
        self._history.append((positions, velocities, visible))
        while len(self._history) < 2:
            self._history.append((positions.copy(), velocities.copy(), visible.copy()))
        return FrameContext(np.stack([x[0] for x in self._history]), np.stack([x[1] for x in self._history]), np.stack([x[2] for x in self._history]), teams), index

    def is_gated(self, now: float) -> bool:
        gated = float(now) < self.next_action_at
        if gated:
            self.gated_ticks += 1
        return gated

    def pop_due_kind(self, now: float) -> str | None:
        if self.pending_kind is None or float(now) < self.next_action_at:
            return None
        kind, self.pending_kind = self.pending_kind, None
        self.executed_counts[kind] += 1
        return kind

    def schedule_after_pass(self, state: MatchAffectiveState, action: PassAction | None) -> None:
        context, index = self.observe(state)
        if action is None:
            return
        receiver_id = action.to_id if action.completed else state.ball.possessor_id
        actor, receiver = index.get(action.from_id, -1), index.get(receiver_id, -1)
        if actor < 0 or receiver < 0:
            return
        opponents = np.flatnonzero(context.visible[-1] & (context.teams != context.teams[receiver]) & (context.teams >= 0))
        if not len(opponents):
            return
        defender = int(opponents[np.argmin(np.linalg.norm(context.positions[-1, opponents] - context.positions[-1, receiver], axis=1))])
        plan = self.router.plan_next_mark(context, ActionRequest(kind="pass", provider=self.provider, actor_index=actor, target_index=receiver, defender_index=defender))
        if plan is None:
            return
        raw_delay = self.max_delay_s if plan.kind == "terminal" else plan.delay_seconds
        delay = float(np.clip(raw_delay, 0.0, self.max_delay_s))
        self.next_action_at = max(self.next_action_at, float(state.clock_seconds) + delay)
        self.pending_kind = plan.kind
        self.plans += 1
        self.kind_counts[plan.kind] += 1
        self.delay_sum_s += delay

    def diagnostics(self) -> dict[str, object]:
        return {"enabled": True, "provider": self.provider, "plans": self.plans, "gated_ticks": self.gated_ticks, "mean_delay_s": self.delay_sum_s / max(1, self.plans), "next_mark_counts": dict(self.kind_counts), "executed_mark_counts": dict(self.executed_counts)}
