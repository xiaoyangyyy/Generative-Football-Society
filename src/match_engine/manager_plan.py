"""Frozen, deterministic in-match manager instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


IN_MATCH_CONDITIONS = {"always", "trailing", "drawing", "leading"}
IN_MATCH_TACTICS = {
    "balanced", "tiki_taka", "gegenpress", "counter_attack",
    "low_block_counter", "direct_vertical",
}
MAX_IN_MATCH_RULES = 5


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
        raise ValueError(f"{field_name} must be a non-empty string up to 128 characters")
    return value.strip()


@dataclass(frozen=True)
class InMatchInstruction:
    rule_id: str
    minute: int
    condition: str = "always"
    tactic: str | None = None
    substitute_off: str | None = None
    substitute_on: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _identifier(self.rule_id, "rule_id"))
        if isinstance(self.minute, bool) or not isinstance(self.minute, int):
            raise ValueError("instruction minute must be an integer")
        if not 1 <= self.minute <= 89:
            raise ValueError("instruction minute must be between 1 and 89")
        if self.condition not in IN_MATCH_CONDITIONS:
            raise ValueError("unsupported in-match condition")
        if self.tactic is not None and self.tactic not in IN_MATCH_TACTICS:
            raise ValueError("unsupported in-match tactic")
        has_off = self.substitute_off is not None
        has_on = self.substitute_on is not None
        if has_off != has_on:
            raise ValueError("substitution requires both off and on player ids")
        if has_off:
            object.__setattr__(
                self, "substitute_off",
                _identifier(self.substitute_off, "substitute_off"),
            )
            object.__setattr__(
                self, "substitute_on",
                _identifier(self.substitute_on, "substitute_on"),
            )
            if self.substitute_off == self.substitute_on:
                raise ValueError("substitution players must be different")
        if self.tactic is None and not has_off:
            raise ValueError("instruction requires a tactic or substitution")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InMatchInstruction":
        if not isinstance(payload, Mapping):
            raise ValueError("in-match instruction must be an object")
        substitution = payload.get("substitution")
        if substitution is not None and not isinstance(substitution, Mapping):
            raise ValueError("instruction substitution must be an object")
        substitution = substitution or {}
        return cls(
            rule_id=payload.get("rule_id"),
            minute=payload.get("minute"),
            condition=payload.get("condition", "always"),
            tactic=payload.get("tactic"),
            substitute_off=substitution.get("off"),
            substitute_on=substitution.get("on"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "minute": self.minute,
            "condition": self.condition,
            "tactic": self.tactic,
            "substitution": (
                {"off": self.substitute_off, "on": self.substitute_on}
                if self.substitute_off is not None else None
            ),
        }


@dataclass(frozen=True)
class InMatchPlan:
    team: str
    instructions: tuple[InMatchInstruction, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "team", _identifier(self.team, "plan team"))
        instructions = tuple(self.instructions)
        object.__setattr__(self, "instructions", instructions)
        if len(instructions) > MAX_IN_MATCH_RULES:
            raise ValueError(f"in-match plan supports at most {MAX_IN_MATCH_RULES} instructions")
        ids = [item.rule_id for item in instructions]
        if len(set(ids)) != len(ids):
            raise ValueError("in-match instruction ids must be unique")
        minutes = [item.minute for item in instructions]
        if minutes != sorted(minutes) or len(set(minutes)) != len(minutes):
            raise ValueError("in-match instruction minutes must be unique and increasing")
        player_ids: list[str] = []
        for item in instructions:
            if item.substitute_off is not None:
                player_ids.extend([item.substitute_off, item.substitute_on or ""])
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("a player may appear in only one planned substitution")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "InMatchPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("in-match plan must be an object")
        raw = payload.get("instructions", [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("in-match instructions must be an array")
        return cls(
            team=payload.get("team"),
            instructions=tuple(InMatchInstruction.from_payload(item) for item in raw),
        )

    def validate_lineup(self, *, starters: Sequence[str], bench: Sequence[str]) -> None:
        starter_ids, bench_ids = set(starters), set(bench)
        for item in self.instructions:
            if item.substitute_off is None:
                continue
            if item.substitute_off not in starter_ids:
                raise ValueError("planned off player must belong to the frozen starting XI")
            if item.substitute_on not in bench_ids:
                raise ValueError("planned on player must belong to the frozen bench")

    @property
    def controls_substitutions(self) -> bool:
        return any(item.substitute_off is not None for item in self.instructions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "team": self.team,
            "instructions": [item.as_dict() for item in self.instructions],
            "controls_substitutions": self.controls_substitutions,
        }


def _condition_matches(condition: str, score_for: int, score_against: int) -> bool:
    return (
        condition == "always"
        or condition == "trailing" and score_for < score_against
        or condition == "drawing" and score_for == score_against
        or condition == "leading" and score_for > score_against
    )


@dataclass
class InMatchPlanRuntime:
    plan: InMatchPlan
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    _resolved: set[str] = field(default_factory=set)

    def step(self, state, *, clock_sec: float, tracker, subs_done: dict) -> list[str]:
        notes: list[str] = []
        team = state.team(self.plan.team)
        opponent = state.away if team is state.home else state.home
        minute = float(clock_sec) / 60.0
        for instruction in self.plan.instructions:
            if instruction.rule_id in self._resolved or minute < instruction.minute:
                continue
            self._resolved.add(instruction.rule_id)
            base = {
                "rule_id": instruction.rule_id,
                "scheduled_minute": instruction.minute,
                "evaluated_minute": minute,
                "condition": instruction.condition,
                "score_for": int(team.score),
                "score_against": int(opponent.score),
                "requested_tactic": instruction.tactic,
                "requested_substitution": (
                    {"off": instruction.substitute_off, "on": instruction.substitute_on}
                    if instruction.substitute_off is not None else None
                ),
            }
            if not _condition_matches(
                instruction.condition, int(team.score), int(opponent.score),
            ):
                self.outcomes.append({
                    **base, "status": "skipped", "reason": "condition_not_met",
                })
                continue
            if instruction.substitute_off is not None:
                from src.match_engine.substitution_engine import (
                    apply_explicit_substitution,
                    validate_explicit_substitution,
                )

                try:
                    off_player, on_player = validate_explicit_substitution(
                        team, instruction.substitute_off,
                        instruction.substitute_on or "",
                    )
                except ValueError as exc:
                    self.outcomes.append({
                        **base, "status": "failed", "reason": str(exc),
                    })
                    continue
                if int(subs_done.get(team.team_id, 0)) >= 5:
                    self.outcomes.append({
                        **base, "status": "failed", "reason": "substitution_limit_reached",
                    })
                    continue
            else:
                off_player = on_player = None
            if instruction.tactic is not None:
                from src.match_engine.tactical_catalog import TACTICAL_PRESETS
                from src.match_engine.tactical_profile import apply_vector_to_team_coach

                apply_vector_to_team_coach(
                    team, dict(TACTICAL_PRESETS[instruction.tactic]),
                )
            sub_note = None
            if off_player is not None and on_player is not None:
                from src.match_engine.substitution_engine import apply_explicit_substitution

                sub_note = apply_explicit_substitution(
                    team, off_player, on_player,
                    minute=minute, tracker=tracker,
                )
                subs_done[team.team_id] = int(subs_done.get(team.team_id, 0)) + 1
            self.outcomes.append({
                **base, "status": "applied", "reason": "condition_met",
                "applied_tactic": instruction.tactic,
                "applied_substitution": (
                    {"off": off_player.player_id, "on": on_player.player_id}
                    if off_player is not None and on_player is not None else None
                ),
            })
            actions = []
            if instruction.tactic:
                actions.append(f"TACTIC {instruction.tactic}")
            if sub_note:
                actions.append(sub_note)
            notes.append(
                f"{int(minute)}' MANAGER {team.team_id}: " + " + ".join(actions)
            )
        return notes

    def diagnostics(self) -> dict[str, Any]:
        pending = [
            item.as_dict() for item in self.plan.instructions
            if item.rule_id not in self._resolved
        ]
        return {
            "available": True,
            "plan": self.plan.as_dict(),
            "outcomes": list(self.outcomes),
            "pending": pending,
            "applied": sum(item["status"] == "applied" for item in self.outcomes),
            "skipped": sum(item["status"] == "skipped" for item in self.outcomes),
            "failed": sum(item["status"] == "failed" for item in self.outcomes),
        }
