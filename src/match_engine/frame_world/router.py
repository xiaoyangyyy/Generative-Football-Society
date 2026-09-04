"""Evidence-gated routing and closed-loop rollout for local frame transitions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math
import numpy as np
import torch
from .schema import BALL_INDEX, ENTITY_COUNT
from .reception import NEXT_ACTIONS, apply_reception_state


@dataclass(frozen=True)
class FrameContext:
    positions: np.ndarray
    velocities: np.ndarray
    visible: np.ndarray
    teams: np.ndarray

    def __post_init__(self):
        if (
            self.positions.ndim != 3
            or self.positions.shape != self.velocities.shape
            or self.positions.shape[1:] != (ENTITY_COUNT, 2)
        ):
            raise ValueError("invalid context positions")
        if self.visible.shape != self.positions.shape[:2] or self.teams.shape != (
            ENTITY_COUNT,
        ):
            raise ValueError("invalid context metadata")


@dataclass(frozen=True)
class ActionRequest:
    kind: str
    provider: str
    actor_index: int = -1
    target_index: int = -1
    defender_index: int = -1
    direction: tuple[float, float] = (0.0, 0.0)
    onset: bool = True


@dataclass(frozen=True)
class RouteDecision:
    action: str
    route: str
    enabled: bool
    reason: str
    affected: tuple[int, ...] = ()


@dataclass(frozen=True)
class TransitionResult:
    positions: np.ndarray
    velocities: np.ndarray
    decisions: tuple[RouteDecision, ...]
    possessor_index: int = -1
    possession_team: int = -1


@dataclass(frozen=True)
class PlannedMark:
    delay_steps: int
    delay_seconds: float
    kind: str
    actor_index: int


@dataclass(frozen=True)
class PlannedMarkInterval:
    lower_steps: int
    delay_steps: int
    upper_steps: int
    lower_seconds: float
    delay_seconds: float
    upper_seconds: float
    coverage: float
    kind: str
    actor_index: int


def _metric(value):
    output = np.asarray(value, np.float32).copy()
    output[..., 0] *= 105
    output[..., 1] *= 68
    return output


def _valid(context, indices):
    return all(0 <= i < ENTITY_COUNT and bool(context.visible[-1, i]) for i in indices)


def pressure_features(context, actor, target):
    pos = context.positions[-1, [actor, target]]
    vel = context.velocities[-1, [actor, target]]
    acc = (
        vel - context.velocities[-2, [actor, target]]
        if len(context.velocities) > 1
        else np.zeros_like(vel)
    )
    delta = _metric(pos[1] - pos[0])
    distance = float(np.linalg.norm(delta))
    unit = delta / max(distance, 1e-6)
    relative = _metric(vel[1] - vel[0])
    closing = -float(relative @ unit)
    geometry = np.array(
        [min(distance / 10, 2), np.clip(closing / 10, -1, 1), unit[0], unit[1]],
        np.float32,
    )
    return np.r_[geometry, vel.ravel(), acc.ravel()].astype(np.float32)


def pass_features(context, receiver, defender):
    slots = [BALL_INDEX, receiver, defender]
    pos = context.positions[-1, slots]
    vel = context.velocities[-1, slots]
    acc = (
        vel - context.velocities[-2, slots]
        if len(context.velocities) > 1
        else np.zeros_like(vel)
    )
    ball, recv, guard = pos
    rr = _metric(recv - ball) / 50
    rd = _metric(guard - ball) / 50
    pair = _metric(guard - recv) / 20
    line = _metric(recv - ball)
    denom = max(float(line @ line), 1e-6)
    point = _metric(guard - ball)
    alpha = np.clip(float(point @ line) / denom, 0, 1)
    lane = float(np.linalg.norm(point - alpha * line)) / 10
    direction = rr / max(float(np.linalg.norm(rr)), 1e-6)
    distance = np.linalg.norm(_metric(recv - ball)) / 50
    return np.r_[
        rr,
        rd,
        pair,
        lane,
        _metric(vel).ravel() / 10,
        _metric(acc).ravel() / 10,
        direction,
        distance,
    ].astype(np.float32)


def temporal_static_features(context, receiver, defender):
    value = pass_features(context, receiver, defender)
    ball, recv, guard = context.positions[-1, [BALL_INDEX, receiver, defender]]
    lane = float(value[6])
    value[7:19] = np.array(
        [
            abs(ball[0] - 0.5),
            abs(ball[1] - 0.5),
            abs(recv[0] - 0.5),
            abs(recv[1] - 0.5),
            abs(guard[0] - 0.5),
            abs(guard[1] - 0.5),
            abs(recv[0] - 0.5),
            abs(ball[0] - 0.5),
            np.linalg.norm(_metric(guard - recv)) / 20,
            lane,
            np.linalg.norm(_metric(recv - ball)) / 50,
            np.linalg.norm(_metric(guard - ball)) / 50,
        ],
        np.float32,
    )
    return value


class ActionTransitionRouter:
    def __init__(
        self,
        pass_models=None,
        pressure_models=None,
        shot_models=None,
        reception_models=None,
        temporal_models=None,
        temporal_thresholds=None,
        temporal_intervals=None,
        horizon_s=0.5,
        max_player_speed_mps=12,
        max_ball_speed_mps=45,
    ):
        self.pass_models = pass_models or {}
        self.pressure_models = pressure_models or {}
        self.shot_models = shot_models or {}
        self.reception_models = reception_models or {}
        self.temporal_models = temporal_models or {}
        self.temporal_thresholds = temporal_thresholds or {}
        self.temporal_intervals = temporal_intervals or {}
        self.horizon_s = float(horizon_s)
        self.max_player_speed_mps = float(max_player_speed_mps)
        self.max_ball_speed_mps = float(max_ball_speed_mps)

    @staticmethod
    def _infer(model, current, velocity, features):
        device = next(model.parameters()).device
        with torch.no_grad():
            result = model.predict(
                torch.from_numpy(current[None].astype(np.float32)).to(device),
                torch.from_numpy(velocity[None].astype(np.float32)).to(device),
                torch.from_numpy(features[None].astype(np.float32)).to(device),
            )
        return result[0].cpu().numpy()

    def _residual(self, context, action, baseline):
        if action.kind == "pass":
            model = self.pass_models.get(action.provider)
            indices = (BALL_INDEX, action.target_index, action.defender_index)
            if model is None:
                return {}, RouteDecision(
                    "pass", "fallback", False, "provider_or_model_unsupported"
                )
            roles = (
                0 <= action.actor_index < ENTITY_COUNT
                and context.teams[action.actor_index] >= 0
                and context.teams[action.actor_index]
                == context.teams[action.target_index]
                and context.teams[action.defender_index] >= 0
                and context.teams[action.defender_index]
                != context.teams[action.actor_index]
                if _valid(context, indices)
                else False
            )
            if not roles:
                return {}, RouteDecision(
                    "pass", "fallback", False, "identity_visibility_or_role_invalid"
                )
            features = pass_features(
                context, action.target_index, action.defender_index
            )
            current = context.positions[-1, list(indices)]
            velocity = context.velocities[-1, list(indices)]
            predicted = self._infer(model, current, velocity, features)
            return {
                slot: predicted[n] - baseline[slot] for n, slot in enumerate(indices)
            }, RouteDecision(
                "pass", "pass_triplet_v85", True, "strong_aligned_triplet", indices
            )
        if action.kind == "pressure":
            model = self.pressure_models.get(action.provider)
            indices = (action.actor_index, action.target_index)
            if model is None or not action.onset:
                return {}, RouteDecision(
                    "pressure", "fallback", False, "onset_or_provider_unsupported"
                )
            roles = (
                _valid(context, indices)
                and context.teams[action.actor_index] >= 0
                and context.teams[action.target_index] >= 0
                and context.teams[action.actor_index]
                != context.teams[action.target_index]
            )
            if not roles:
                return {}, RouteDecision(
                    "pressure", "fallback", False, "identity_visibility_or_role_invalid"
                )
            features = pressure_features(context, *indices)
            current = context.positions[-1, list(indices)]
            velocity = context.velocities[-1, list(indices)]
            predicted = self._infer(model, current, velocity, features)
            return {
                slot: predicted[n] - baseline[slot] for n, slot in enumerate(indices)
            }, RouteDecision(
                "pressure", "pressure_pair_v84", True, "strong_visible_onset", indices
            )
        if action.kind == "shot":
            model = self.shot_models.get(action.provider)
            norm = float(np.linalg.norm(action.direction))
            if model is None or norm < 0.5:
                return {}, RouteDecision(
                    "shot", "fallback", False, "direction_or_model_unsupported"
                )
            device = next(model.parameters()).device
            semantic = np.array(
                [[0, 1, 0, action.direction[0] / norm, action.direction[1] / norm]],
                np.float32,
            )
            actors = np.full((1, 3), -1, np.int64)
            targets = np.full((1, 3), -1, np.int64)
            actors[0, 1] = action.actor_index
            with torch.no_grad():
                predicted, _ = model.predict(
                    torch.from_numpy(context.positions[None].astype(np.float32)).to(
                        device
                    ),
                    torch.from_numpy(context.velocities[None].astype(np.float32)).to(
                        device
                    ),
                    torch.from_numpy(context.visible[None].astype(bool)).to(device),
                    torch.from_numpy(context.teams[None].astype(np.int64)).to(device),
                    torch.from_numpy(semantic).to(device),
                    torch.from_numpy(actors).to(device),
                    torch.from_numpy(targets).to(device),
                )
            value = predicted[0, BALL_INDEX].cpu().numpy() - baseline[BALL_INDEX]
            return {BALL_INDEX: value}, RouteDecision(
                "shot", "semantic_shot_v83", True, "validated_direction", (BALL_INDEX,)
            )
        return {}, RouteDecision(action.kind, "fallback", False, "unknown_action")

    def transition(self, context: FrameContext, actions: Iterable[ActionRequest] = ()):
        current = context.positions[-1]
        velocity = context.velocities[-1]
        baseline = np.clip(current + velocity * self.horizon_s, -0.15, 1.15)
        residuals = {}
        decisions = []
        for action in actions:
            updates, decision = self._residual(context, action, baseline)
            decisions.append(decision)
            for index, value in updates.items():
                residuals[index] = (
                    value if index == BALL_INDEX else residuals.get(index, 0) + value
                )
        predicted = baseline.copy()
        for index, value in residuals.items():
            predicted[index] = baseline[index] + value
        displacement = _metric(predicted - current)
        distance = np.linalg.norm(displacement, axis=1)
        limits = np.full(
            ENTITY_COUNT, self.max_player_speed_mps * self.horizon_s, np.float32
        )
        limits[BALL_INDEX] = self.max_ball_speed_mps * self.horizon_s
        scale = np.minimum(1, limits / np.maximum(distance, 1e-6))
        displacement *= scale[:, None]
        predicted = current + displacement / np.array([105, 68], np.float32)
        predicted = np.clip(predicted, -0.15, 1.15).astype(np.float32)
        next_velocity = (predicted - current) / self.horizon_s
        candidates = np.flatnonzero(context.visible[-1, :BALL_INDEX])
        possessor = -1
        if len(candidates):
            distances = np.linalg.norm(
                _metric(predicted[candidates] - predicted[BALL_INDEX]), axis=1
            )
            nearest = int(np.argmin(distances))
            possessor = int(candidates[nearest]) if distances[nearest] <= 2 else -1
        possession = int(context.teams[possessor]) if possessor >= 0 else -1
        return TransitionResult(
            predicted,
            next_velocity.astype(np.float32),
            tuple(decisions),
            possessor,
            possession,
        )

    def resolve_reception(self, context: FrameContext, action: ActionRequest):
        model = self.reception_models.get(action.provider)
        if (
            action.kind != "pass"
            or model is None
            or not _valid(
                context, (BALL_INDEX, action.target_index, action.defender_index)
            )
        ):
            return None
        features = pass_features(context, action.target_index, action.defender_index)
        device = next(model.parameters()).device
        with torch.no_grad():
            completion, next_logits = model(torch.from_numpy(features[None]).to(device))
        complete = bool(torch.sigmoid(completion)[0] >= 0.5)
        next_action = NEXT_ACTIONS[int(next_logits[0].argmax())]
        return apply_reception_state(
            complete,
            action.target_index,
            action.defender_index,
            context.teams,
            next_action,
        )

    def plan_next_mark(self, context: FrameContext, action: ActionRequest):
        model = self.temporal_models.get(action.provider)
        if (
            action.kind != "pass"
            or model is None
            or not _valid(
                context, (BALL_INDEX, action.target_index, action.defender_index)
            )
        ):
            return None
        static_contract = getattr(model.cfg, "direct_subtype", False) or hasattr(
            model.cfg, "components"
        )
        features = (
            temporal_static_features(
                context, action.target_index, action.defender_index
            )
            if static_contract
            else pass_features(context, action.target_index, action.defender_index)
        )
        device = next(model.parameters()).device
        thresholds = self.temporal_thresholds.get(action.provider, (0.5, 0.5))
        with torch.no_grad():
            timing, kinds = model.predict_mark(
                torch.from_numpy(features[None]).to(device), *thresholds
            )
        if hasattr(model.cfg, "components"):
            seconds = float(timing[0])
            step = max(1, int(math.ceil(seconds / self.horizon_s)))
        else:
            step = int(timing[0])
            seconds = step * model.cfg.bin_seconds
        kind = ("pass", "shot", "terminal")[int(kinds[0])]
        actor = action.target_index if kind != "terminal" else -1
        return PlannedMark(step, seconds, kind, actor)

    def plan_next_mark_interval(
        self, context: FrameContext, action: ActionRequest, coverage: float = 0.9
    ):
        plan = self.plan_next_mark(context, action)
        radii = self.temporal_intervals.get(action.provider, {})
        selected = (
            radii.get(plan.kind, radii.get("global", radii)) if plan is not None else {}
        )
        radius = selected.get(coverage, selected.get(str(coverage)))
        if plan is None or radius is None or not 0 < coverage < 1 or float(radius) < 0:
            return None
        lower = max(0.0, plan.delay_seconds - float(radius))
        upper = min(5.0, plan.delay_seconds + float(radius))
        return PlannedMarkInterval(
            int(math.floor(lower / self.horizon_s)),
            plan.delay_steps,
            max(1, int(math.ceil(upper / self.horizon_s))),
            lower,
            plan.delay_seconds,
            upper,
            float(coverage),
            plan.kind,
            plan.actor_index,
        )

    @staticmethod
    def _apply_planned_mark(context: FrameContext, kind: str, actor_index: int):
        positions = context.positions.copy()
        velocities = context.velocities.copy()
        if (
            kind != "terminal"
            and 0 <= actor_index < ENTITY_COUNT
            and context.visible[-1, actor_index]
        ):
            positions[-1, BALL_INDEX] = positions[-1, actor_index]
            velocities[-1, BALL_INDEX] = velocities[-1, actor_index]
        else:
            velocities[-1, BALL_INDEX] = 0
        return FrameContext(
            positions, velocities, context.visible.copy(), context.teams
        )

    def rollout_event_driven_interval(
        self,
        context: FrameContext,
        action: ActionRequest,
        steps: int,
        bound: str = "median",
        coverage: float = 0.9,
    ):
        interval = self.plan_next_mark_interval(context, action, coverage)
        if interval is None:
            return self.rollout_event_driven(context, action, steps)
        selected = {
            "lower": interval.lower_steps,
            "median": interval.delay_steps,
            "upper": interval.upper_steps,
        }
        if bound not in selected:
            raise ValueError("bound must be lower, median, or upper")
        trigger = selected[bound]
        positions = []
        decisions = []
        state = context
        for step in range(steps):
            if step == trigger:
                state = self._apply_planned_mark(
                    state, interval.kind, interval.actor_index
                )
            result = self.transition(state, [action] if step == 0 else ())
            positions.append(result.positions)
            decisions.extend(result.decisions)
            state = FrameContext(
                np.concatenate([state.positions[1:], result.positions[None]], 0),
                np.concatenate([state.velocities[1:], result.velocities[None]], 0),
                state.visible.copy(),
                state.teams,
            )
        return np.stack(positions), interval, tuple(decisions)

    def rollout_event_driven(
        self, context: FrameContext, action: ActionRequest, steps: int
    ):
        plan = self.plan_next_mark(context, action)
        positions = []
        decisions = []
        state = context
        for step in range(steps):
            if plan is not None and step == plan.delay_steps:
                state = self._apply_planned_mark(state, plan.kind, plan.actor_index)
            result = self.transition(state, [action] if step == 0 else ())
            positions.append(result.positions)
            decisions.extend(result.decisions)
            state = FrameContext(
                np.concatenate([state.positions[1:], result.positions[None]], 0),
                np.concatenate([state.velocities[1:], result.velocities[None]], 0),
                state.visible.copy(),
                state.teams,
            )
        return np.stack(positions), plan, tuple(decisions)

    def rollout(self, context: FrameContext, steps: int, schedule=None):
        schedule = schedule or {}
        positions = []
        decisions = []
        state = context
        for step in range(steps):
            result = self.transition(state, schedule.get(step, ()))
            positions.append(result.positions)
            decisions.extend(result.decisions)
            state = FrameContext(
                np.concatenate([state.positions[1:], result.positions[None]], 0),
                np.concatenate([state.velocities[1:], result.velocities[None]], 0),
                state.visible.copy(),
                state.teams,
            )
        return np.stack(positions), tuple(decisions)
