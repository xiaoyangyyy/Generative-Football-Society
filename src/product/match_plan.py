"""Validated, auditable match experiences exposed by GFS Studio."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


NATIVE_TACTIC = "team_identity"
PLAYABLE_TACTICS: dict[str, dict[str, str]] = {
    NATIVE_TACTIC: {
        "label": "球队原生体系",
        "description": "保留数据与教练画像推导的原始战术。",
    },
    "balanced": {
        "label": "均衡控制",
        "description": "在控球、风险、压迫和纵深之间保持平衡。",
    },
    "tiki_taka": {
        "label": "短传控球",
        "description": "强调短传组织、位置轮转与控球耐心。",
    },
    "gegenpress": {
        "label": "高位反抢",
        "description": "用高压迫、高节奏和丢球反抢争夺主动。",
    },
    "counter_attack": {
        "label": "快速反击",
        "description": "降低持球依赖，强调转换速度和纵向推进。",
    },
    "low_block_counter": {
        "label": "低位防反",
        "description": "保持低位紧凑结构，并在夺回球权后快速反击。",
    },
    "direct_vertical": {
        "label": "直接纵向",
        "description": "减少横向传递，优先快速向前和直塞。",
    },
}
MATCH_EXPERIENCES = {
    "observational", "tactical_lab", "world_model_lab", "season_manager",
}
WORLD_MODEL_POLICIES = {"mode_default", "predict_only", "action_policy"}


@dataclass(frozen=True)
class MatchPlan:
    """One reproducible Studio request with an explicit inference boundary."""

    experience: str = "observational"
    home_tactic: str = NATIVE_TACTIC
    away_tactic: str = NATIVE_TACTIC
    reuse_last_seed: bool = False
    world_model_policy: str = "mode_default"
    world_model_branch_at_sec: float | None = None

    def __post_init__(self) -> None:
        if self.experience not in MATCH_EXPERIENCES:
            raise ValueError("unsupported match experience")
        for side, tactic in (
            ("home", self.home_tactic), ("away", self.away_tactic),
        ):
            if tactic not in PLAYABLE_TACTICS:
                raise ValueError(f"unsupported {side} tactic")
        if not isinstance(self.reuse_last_seed, bool):
            raise ValueError("reuse_last_seed must be boolean")
        if self.world_model_policy not in WORLD_MODEL_POLICIES:
            raise ValueError("unsupported world-model policy")
        if self.world_model_branch_at_sec is not None:
            if (
                isinstance(self.world_model_branch_at_sec, bool)
                or not isinstance(self.world_model_branch_at_sec, (int, float))
                or not math.isfinite(float(self.world_model_branch_at_sec))
                or not 0.0 <= float(self.world_model_branch_at_sec) <= 5400.0
            ):
                raise ValueError("world-model branch time must be between 0 and 5400 seconds")
            if self.experience != "world_model_lab":
                raise ValueError("branch time is confined to world_model_lab")
        interventions = (
            self.home_tactic != NATIVE_TACTIC
            or self.away_tactic != NATIVE_TACTIC
        )
        if self.experience == "observational" and interventions:
            raise ValueError("observational matches cannot override tactics")
        if self.experience == "observational" and self.reuse_last_seed:
            raise ValueError(
                "seed reuse is only available in tactical_lab or world_model_lab"
            )
        if self.experience == "tactical_lab" and not interventions:
            raise ValueError("tactical_lab requires at least one tactical intervention")
        if self.experience == "world_model_lab":
            if self.world_model_policy not in {"predict_only", "action_policy"}:
                raise ValueError(
                    "world_model_lab requires an explicit world-model policy"
                )
        elif self.world_model_policy != "mode_default":
            raise ValueError(
                "world-model policy overrides are confined to world_model_lab"
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "MatchPlan":
        raw = payload or {}
        if not isinstance(raw, Mapping):
            raise ValueError("plan must be an object")
        return cls(
            experience=raw.get("experience", "observational"),
            home_tactic=raw.get("home_tactic", NATIVE_TACTIC),
            away_tactic=raw.get("away_tactic", NATIVE_TACTIC),
            reuse_last_seed=raw.get("reuse_last_seed", False),
            world_model_policy=raw.get("world_model_policy", "mode_default"),
            world_model_branch_at_sec=raw.get("world_model_branch_at_sec"),
        )

    def validate_for_mode(self, studio_mode: str, *, context: str = "standalone") -> None:
        if self.experience == "tactical_lab" and studio_mode not in {
            "research", "cognitive",
        }:
            raise ValueError("tactical_lab requires research or cognitive mode")
        if self.experience == "world_model_lab" and studio_mode != "research":
            raise ValueError("world_model_lab requires deterministic research mode")
        if self.experience == "season_manager" and context != "season":
            raise ValueError("season_manager is only available inside a season fixture")

    @property
    def score_path(self) -> str:
        return (
            "physics_official"
            if self.experience in {
                "tactical_lab", "world_model_lab", "season_manager",
            }
            else "macro_replay"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experience": self.experience,
            "home_tactic": self.home_tactic,
            "away_tactic": self.away_tactic,
            "reuse_last_seed": self.reuse_last_seed,
            "world_model_policy": self.world_model_policy,
            "world_model_branch_at_sec": self.world_model_branch_at_sec,
            "score_path": self.score_path,
            "claim_boundary": (
                "gameplay intervention only; no causal or real-world claim"
                if self.experience == "season_manager" else
                "shared-seed policy contrast only; not a population or real-world causal effect"
                if self.experience == "world_model_lab" else
                "single_run_descriptive_only; use a shared-seed pair for tactical attribution"
            ),
        }


@dataclass(frozen=True)
class PairedMatchPlan:
    """One explicit same-seed baseline/treatment tactical contrast."""

    baseline_home_tactic: str
    baseline_away_tactic: str
    treatment_home_tactic: str
    treatment_away_tactic: str
    seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**31 - 1
        ):
            raise ValueError("paired match seed must be a 32-bit non-negative integer")
        baseline = self.baseline_plan()
        treatment = self.treatment_plan()
        changed_sides = [
            side for side in ("home", "away")
            if getattr(baseline, f"{side}_tactic")
            != getattr(treatment, f"{side}_tactic")
        ]
        if len(changed_sides) != 1:
            raise ValueError("paired match requires exactly one changed tactical side")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PairedMatchPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("paired match plan must be an object")
        baseline = payload.get("baseline") or {}
        treatment = payload.get("treatment") or {}
        if not isinstance(baseline, Mapping) or not isinstance(treatment, Mapping):
            raise ValueError("paired baseline and treatment must be objects")
        return cls(
            baseline_home_tactic=baseline.get("home_tactic"),
            baseline_away_tactic=baseline.get("away_tactic"),
            treatment_home_tactic=treatment.get("home_tactic"),
            treatment_away_tactic=treatment.get("away_tactic"),
            seed=payload.get("seed"),
        )

    @property
    def focus_side(self) -> str:
        return "home" if (
            self.baseline_home_tactic != self.treatment_home_tactic
        ) else "away"

    def baseline_plan(self) -> MatchPlan:
        return MatchPlan(
            experience="tactical_lab",
            home_tactic=self.baseline_home_tactic,
            away_tactic=self.baseline_away_tactic,
        )

    def treatment_plan(self) -> MatchPlan:
        return MatchPlan(
            experience="tactical_lab",
            home_tactic=self.treatment_home_tactic,
            away_tactic=self.treatment_away_tactic,
            reuse_last_seed=True,
        )

    def validate_for_mode(self, studio_mode: str) -> None:
        self.baseline_plan().validate_for_mode(studio_mode)
        self.treatment_plan().validate_for_mode(studio_mode)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "focus_side": self.focus_side,
            "baseline": self.baseline_plan().as_dict(),
            "treatment": self.treatment_plan().as_dict(),
            "claim_boundary": (
                "single shared-seed contrast for this fixture only; "
                "not a population effect or significance test"
            ),
        }


@dataclass(frozen=True)
class WorldModelForkPlan:
    """One same-seed predict-only/action-policy product contrast."""

    home_tactic: str
    away_tactic: str
    seed: int
    branch_at_sec: float = 2700.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**31 - 1
        ):
            raise ValueError("world-model fork seed must be a 32-bit non-negative integer")
        for side, tactic in (
            ("home", self.home_tactic), ("away", self.away_tactic),
        ):
            if tactic not in PLAYABLE_TACTICS:
                raise ValueError(f"unsupported {side} tactic")
        if (
            isinstance(self.branch_at_sec, bool)
            or not isinstance(self.branch_at_sec, (int, float))
            or not math.isfinite(float(self.branch_at_sec))
            or not 0.0 <= float(self.branch_at_sec) <= 5400.0
        ):
            raise ValueError("world-model fork branch time must be between 0 and 5400 seconds")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WorldModelForkPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("world-model fork plan must be an object")
        return cls(
            home_tactic=payload.get("home_tactic", NATIVE_TACTIC),
            away_tactic=payload.get("away_tactic", NATIVE_TACTIC),
            seed=payload.get("seed"),
            branch_at_sec=payload.get("branch_at_sec", 2700.0),
        )

    def baseline_plan(self) -> MatchPlan:
        return MatchPlan(
            experience="world_model_lab",
            home_tactic=self.home_tactic,
            away_tactic=self.away_tactic,
            world_model_policy="predict_only",
            world_model_branch_at_sec=float(self.branch_at_sec),
        )

    def treatment_plan(self) -> MatchPlan:
        return MatchPlan(
            experience="world_model_lab",
            home_tactic=self.home_tactic,
            away_tactic=self.away_tactic,
            reuse_last_seed=True,
            world_model_policy="action_policy",
            world_model_branch_at_sec=float(self.branch_at_sec),
        )

    def validate_for_mode(self, studio_mode: str) -> None:
        self.baseline_plan().validate_for_mode(studio_mode)
        self.treatment_plan().validate_for_mode(studio_mode)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "home_tactic": self.home_tactic,
            "away_tactic": self.away_tactic,
            "branch_at_sec": float(self.branch_at_sec),
            "baseline_policy": "predict_only",
            "treatment_policy": "action_policy",
            "claim_boundary": (
                "single shared-seed simulator policy contrast only; not a population "
                "effect, significance test, real-football causal effect, or promotion"
            ),
        }


@dataclass(frozen=True)
class WorldModelForkSetPlan:
    """A fixed set of branch-time contrasts under identical controls."""

    home_tactic: str
    away_tactic: str
    seed: int
    branch_times_sec: tuple[float, ...] = (1800.0, 2700.0, 3600.0)

    def __post_init__(self) -> None:
        WorldModelForkPlan(
            home_tactic=self.home_tactic,
            away_tactic=self.away_tactic,
            seed=self.seed,
        )
        raw = tuple(self.branch_times_sec)
        if not 2 <= len(raw) <= 4:
            raise ValueError("world-model fork set requires 2 to 4 branch times")
        normalized: list[float] = []
        for value in raw:
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 5400.0
            ):
                raise ValueError("fork-set branch times must be between 0 and 5400 seconds")
            normalized.append(float(value))
        if normalized != sorted(normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("fork-set branch times must be unique and increasing")
        object.__setattr__(self, "branch_times_sec", tuple(normalized))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WorldModelForkSetPlan":
        if not isinstance(payload, Mapping):
            raise ValueError("world-model fork set plan must be an object")
        return cls(
            home_tactic=payload.get("home_tactic", NATIVE_TACTIC),
            away_tactic=payload.get("away_tactic", NATIVE_TACTIC),
            seed=payload.get("seed"),
            branch_times_sec=tuple(payload.get("branch_times_sec") or ()),
        )

    def fork_plan(self, branch_at_sec: float) -> WorldModelForkPlan:
        if float(branch_at_sec) not in self.branch_times_sec:
            raise ValueError("branch time is outside the frozen fork set")
        return WorldModelForkPlan(
            home_tactic=self.home_tactic,
            away_tactic=self.away_tactic,
            seed=self.seed,
            branch_at_sec=float(branch_at_sec),
        )

    def validate_for_mode(self, studio_mode: str) -> None:
        for branch_at_sec in self.branch_times_sec:
            self.fork_plan(branch_at_sec).validate_for_mode(studio_mode)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "home_tactic": self.home_tactic,
            "away_tactic": self.away_tactic,
            "branch_times_sec": list(self.branch_times_sec),
            "fixed_scenario_budget": len(self.branch_times_sec),
            "baseline_policy": "predict_only",
            "treatment_policy": "action_policy",
            "analysis_policy": "descriptive_timing_sensitivity_no_ranking",
            "claim_boundary": (
                "fixed fixture, seed, tactics, checkpoint and branch-time set; "
                "simulator contrasts only, with no best-time recommendation, "
                "population inference, outcome causality, real-football causality, "
                "or promotion authorization"
            ),
        }


def playable_tactic_catalog() -> list[dict[str, str]]:
    return [
        {"id": tactic_id, **metadata}
        for tactic_id, metadata in PLAYABLE_TACTICS.items()
    ]
