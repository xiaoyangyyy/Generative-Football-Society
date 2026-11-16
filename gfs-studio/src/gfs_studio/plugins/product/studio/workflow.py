import copy

from gfs_core.contracts import PluginContext, PluginMetadata
from gfs_core.kernel.identity import content_identity
from gfs_studio.contracts.product import DecisionCase

from .candidates import CandidateEvaluator, candidate_record


class ProductWorkflows:
    def __init__(self, project_root, match_workflow=None, *, overview=None, closer=None, policy_assessor=None, trace_codec=None) -> None:
        self.project_root = project_root
        self.match_workflow = match_workflow
        self.policy_assessor = policy_assessor
        self.trace_codec = trace_codec
        self._overview = overview
        self._closer = closer

    def overview(self) -> dict:
        if self._overview is None:
            return {"deployment": "loopback_only", "overview": "not_configured"}
        return {**self._overview(), "deployment": "loopback_only"}

    def close(self) -> None:
        if self._closer is not None:
            self._closer()
            self._closer = None

    def micro_summary(
        self, home: str, away: str, seed: int, *, home_tactics=None,
        trace_sink=None, before_tick=None,
    ):
        if self.match_workflow is None:
            raise RuntimeError("match workflow capability is not configured")
        if before_tick is not None:
            downstream = trace_sink
            def trace_sink(record):
                before_tick()
                if downstream is not None:
                    downstream(record)
        return self.match_workflow.run(
            home,
            away,
            seed=seed,
            fast=True,
            home_tactics=home_tactics,
            trace_sink=trace_sink,
        )

    def micro_match(self, home: str, away: str, seed: int, *, before_tick=None) -> dict:
        summary = self.micro_summary(home, away, seed, before_tick=before_tick)
        return {
            "home": summary.home_team,
            "away": summary.away_team,
            "score": [summary.goals_micro_home, summary.goals_micro_away],
            "xg": [summary.micro_xg_home, summary.micro_xg_away],
            "passes": [summary.passes_home, summary.passes_away],
            "pass_completion": [
                summary.pass_completion_home,
                summary.pass_completion_away,
            ],
            "shots": [summary.shots_home, summary.shots_away],
            "possession_home": summary.possession_home,
            "ticks": summary.ticks,
        }

    def decision_case(self, home: str, away: str, seed: int, *, observation=None) -> DecisionCase:
        candidate_ids = ("preserve_current_vector", "attacking_pressure")
        case_id = content_identity({
            "home": home,
            "away": away,
            "seed": seed,
            "candidate_ids": candidate_ids,
            "observation_identity": content_identity(observation) if observation is not None else None,
        })
        return DecisionCase(case_id, home, away, seed, candidate_ids)

    def compare_tactics(self, home: str, away: str, seed: int, *, seeds=None) -> dict:
        seed_set = list(seeds) if seeds is not None else [seed, seed + 1]
        if not seed_set or any(isinstance(value, bool) or not isinstance(value, int) for value in seed_set):
            raise ValueError("candidate seeds must be integers")
        definitions = (
            ("preserve_current_vector", {}),
            ("attacking_pressure", {
                "pressing_intensity": 0.75,
                "risk_budget": 0.75,
                "line_height": 0.7,
                "tempo": 0.7,
            }),
        )
        observations = []
        baseline = None
        initial_states = []

        def capture_initial(record):
            if not initial_states:
                initial_states.append(copy.deepcopy(record.state))

        for value in seed_set:
            for name, tactics in definitions:
                capture = capture_initial if baseline is None else None
                summary = self.micro_summary(
                    home,
                    away,
                    value,
                    home_tactics=tactics,
                    trace_sink=capture,
                )
                if name == "preserve_current_vector" and baseline is None:
                    baseline = summary
                observations.append(candidate_record(name, summary, tactics, value))
        evaluation = CandidateEvaluator().evaluate(observations)
        if self.policy_assessor is not None and self.trace_codec is not None:
            evaluation["model_assessments"] = self._model_assessments(
                home, away, seed_set[0], initial_states[0],
            )
        return {
            "baseline": baseline,
            "candidates": observations,
            "evaluation": evaluation,
        }

    def _model_assessments(self, home, away, seed, state) -> list[dict]:
        observation = self.trace_codec.observation(state)
        case = self.decision_case(home, away, seed, observation=observation)
        assessments = []
        for candidate_id in case.candidate_ids:
            kind = "pass" if candidate_id == "attacking_pressure" else "hold"
            action = self.trace_codec.action(kind, state.ball.position, 30.0)
            assessments.append(self.policy_assessor.assess_case(
                case, observation, action,
            ))
        return assessments


class ProductStudioPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.product.studio",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.product",
        provides=("studio-workflows",),
        requires=("gfs.core.match.workflow",),
    )

    def __init__(self, project_root, workflows=None) -> None:
        self._project_root = project_root
        self._workflows = workflows
        self._unsubscribe = None

    def activate(self, context: PluginContext) -> None:
        if self._workflows is None:
            self._workflows = ProductWorkflows(self._project_root)
        if self._workflows.match_workflow is None:
            self._workflows.match_workflow = context.require("match-workflow")
        if self._workflows.policy_assessor is None:
            self._workflows.policy_assessor = context.find("policies")
        if self._workflows.trace_codec is None:
            self._workflows.trace_codec = context.find("trace-codec")
        context.provide("studio-workflows", self._workflows)
        self._unsubscribe = context.subscribe(
            "capability.provided", self._capability_provided,
        )

    def _capability_provided(self, payload) -> None:
        if self._workflows is None:
            return
        if payload["capability"] == "policies" and self._workflows.policy_assessor is None:
            self._workflows.policy_assessor = payload["service"]
        if payload["capability"] == "trace-codec" and self._workflows.trace_codec is None:
            self._workflows.trace_codec = payload["service"]

    def deactivate(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._workflows is not None:
            self._workflows.close()
            self._workflows = None
