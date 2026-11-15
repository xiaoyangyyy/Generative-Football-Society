from __future__ import annotations

import numpy as np

from gfs_core.contracts import PluginContext, PluginMetadata

from .current import CurrentSimulationEvaluator


class CalibrationService:
    def __init__(self, evidence, evaluator, match_workflow) -> None:
        self._evidence = evidence
        self._evaluator = evaluator
        self._current = CurrentSimulationEvaluator(match_workflow, evidence)

    def evaluate_current(self, **options) -> dict:
        return self._current.evaluate(**options)

    def evaluate_candidate(self, model, dataset) -> dict:
        return {
            "transition_prediction": self._evaluator.evaluate(model, dataset),
            "probability_calibration": "not_evaluated_without_event_outcomes",
            "new_training": False,
        }

    def probability_report(self, probabilities, outcomes, *, bins: int = 10) -> dict:
        predicted = np.asarray(probabilities, dtype=np.float64)
        observed = np.asarray(outcomes, dtype=np.float64)
        if predicted.ndim != 1 or predicted.shape != observed.shape or not predicted.size:
            raise ValueError("calibration requires matching non-empty one-dimensional inputs")
        if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
            raise ValueError("calibration inputs must be finite")
        if ((predicted < 0) | (predicted > 1)).any():
            raise ValueError("probabilities must be between zero and one")
        if not np.isin(observed, [0.0, 1.0]).all():
            raise ValueError("calibration outcomes must be binary")
        if not 1 <= bins <= len(predicted):
            raise ValueError("calibration bin count is invalid")
        groups = np.minimum((predicted * bins).astype(int), bins - 1)
        calibration_bins = []
        error = 0.0
        for index in range(bins):
            selected = groups == index
            count = int(selected.sum())
            if not count:
                continue
            confidence = float(predicted[selected].mean())
            frequency = float(observed[selected].mean())
            error += count / len(predicted) * abs(confidence - frequency)
            calibration_bins.append({
                "bin": index,
                "count": count,
                "confidence": confidence,
                "observed_frequency": frequency,
            })
        clipped = np.clip(predicted, 1e-12, 1 - 1e-12)
        return {
            "samples": len(predicted),
            "brier_score": float(np.mean((predicted - observed) ** 2)),
            "log_loss": float(-np.mean(
                observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped)
            )),
            "expected_calibration_error": error,
            "bins": calibration_bins,
            "scope": "descriptive_probability_calibration_no_causal_claim",
        }

    def history(self) -> dict:
        return {
            "plan": self._evidence.read_frozen("m0_calibration_plan_v1.json"),
            "evidence": "frozen_historical",
            "new_runs": 0,
        }


class CalibrationPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.research.calibration",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.research",
        provides=("calibration",),
        requires=(
            "gfs.studio.evidence.receipts",
            "gfs.studio.training.evaluation",
            "gfs.core.match.workflow",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("calibration", CalibrationService(
            context.require("receipts"),
            context.require("model-evaluation"),
            context.require("match-workflow"),
        ))

    def deactivate(self) -> None:
        pass
