from __future__ import annotations

from collections import Counter
from datetime import datetime

import numpy as np

from gfs_core.contracts import PluginContext, PluginMetadata


class ScoringService:
    def __init__(self, sessions, evidence) -> None:
        self.sessions = sessions
        self.protocol = sessions.protocol
        self.seal = evidence.read_frozen("product_value_scoring_seal_v1.json")

    def score_receipt(self, receipt: dict) -> dict:
        if receipt["protocol_id"] != self.protocol["protocol_id"]:
            raise ValueError("receipt protocol mismatch")
        key = self.seal["scoring_keys"].get(receipt["case_pack"])
        if key is None:
            raise ValueError("receipt case pack is unknown")
        answers = receipt["answers"]
        correct = {field: answers.get(field) == value for field, value in key.items()}
        started = datetime.fromisoformat(receipt["started_at"])
        submitted = datetime.fromisoformat(receipt["submitted_at"])
        elapsed = (submitted - started).total_seconds()
        maximum = self.protocol["session_execution"]["server_authoritative_deadline_seconds"]
        if not 0 <= elapsed <= maximum:
            raise ValueError("receipt timing is invalid")
        critical_values = self.protocol["session_execution"]["critical_error_policy"]["values"]
        return {
            "registration_id": receipt["registration_id"],
            "condition": receipt["condition"],
            "case_pack": receipt["case_pack"],
            "correct": correct,
            "accuracy": sum(correct.values()) / len(key),
            "penalized_seconds": elapsed if all(correct.values()) else float(maximum),
            "critical_error": answers.get("supported_claim") in critical_values,
        }

    def score_participant(self, participant_id: str) -> dict:
        record = next(
            (item for item in self.sessions.records() if item["participant_id"] == participant_id),
            None,
        )
        if record is None:
            raise KeyError(participant_id)
        conditions = [self.score_receipt(receipt) for receipt in record["receipts"]]
        by_condition = {item["condition"]: item for item in conditions}
        complete = (
            record["state"] == "completed"
            and record["moderator_attested"]
            and set(by_condition) == set(self.protocol["design"]["conditions"])
            and len(conditions) == 2
        )
        result = {
            "registration_id": record["registration_id"],
            "role": record["role"],
            "sequence": record["sequence"],
            "valid_for_analysis": complete,
            "moderator_attested": record["moderator_attested"],
            "conditions": conditions,
        }
        if complete:
            studio = by_condition["gfs_studio"]
            baseline = by_condition["manual_baseline"]
            result.update({
                "penalized_time_ratio": studio["penalized_seconds"] / max(baseline["penalized_seconds"], 1e-12),
                "accuracy_delta": studio["accuracy"] - baseline["accuracy"],
                "studio_critical_error": studio["critical_error"],
            })
        return result

    def evaluate(self) -> dict:
        registered = self.sessions.records()
        results = [self.score_participant(item["participant_id"]) for item in registered]
        valid = [item for item in results if item["valid_for_analysis"]]
        population = self.protocol["population"]
        role_counts = Counter(item["role"] for item in valid)
        sequence_counts = Counter(item["sequence"] for item in valid)
        role_sequences = Counter((item["role"], item["sequence"]) for item in valid)
        complete = (
            bool(valid)
            and len(valid) == len(registered)
            and len(valid) >= population["minimum_valid_participants"]
            and all(role_counts[role] >= population["minimum_per_role"] for role in population["target_roles"])
            and all(sequence_counts[sequence] >= population["minimum_per_sequence"] for sequence in self.protocol["allocation"]["sequences"])
            and all(role_sequences[role, sequence] >= population["minimum_per_role_sequence"] for role in population["target_roles"] for sequence in self.protocol["allocation"]["sequences"])
        )
        response = {
            "registered_participants": len(registered),
            "valid_participants": len(valid),
            "study_complete": complete,
            "user_value_validated": False,
            "results": results,
            "scope": "preregistered_participant_study_not_product_release_proof",
        }
        if not complete:
            return response
        ratios = np.asarray([item["penalized_time_ratio"] for item in valid])
        accuracy = np.asarray([item["accuracy_delta"] for item in valid])
        endpoint = self.protocol["primary_endpoint"]
        ratio_low, ratio_high = _bootstrap(ratios, endpoint["bootstrap_seed"], endpoint["bootstrap_draws"], log=True)
        accuracy_low, _ = _bootstrap(accuracy, endpoint["bootstrap_seed"], endpoint["bootstrap_draws"])
        critical_errors = sum(item["studio_critical_error"] for item in valid)
        response.update({
            "geometric_mean_time_ratio": float(np.exp(np.log(ratios).mean())),
            "time_ratio_ci95": [ratio_low, ratio_high],
            "mean_accuracy_delta": float(accuracy.mean()),
            "accuracy_delta_ci95_low": accuracy_low,
            "studio_critical_errors": critical_errors,
            "user_value_validated": (
                ratio_high <= endpoint["threshold"]
                and accuracy_low >= self.protocol["safety_and_quality"]["accuracy_noninferiority_margin"]
                and critical_errors == 0
            ),
        })
        return response


class ScoringPlugin:
    metadata = PluginMetadata(
        id="gfs.studio.participants.scoring",
        version="0.1.0",
        api=1,
        contract="gfs_studio.contracts.participants",
        provides=("participant-scoring",),
        requires=(
            "gfs.studio.participants.sessions",
            "gfs.studio.evidence.receipts",
        ),
    )

    def activate(self, context: PluginContext) -> None:
        context.provide("participant-scoring", ScoringService(
            context.require("participant-sessions"), context.require("receipts"),
        ))

    def deactivate(self) -> None:
        pass


def _bootstrap(values, seed: int, draws: int, *, log: bool = False) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices]
    means = np.exp(np.log(samples).mean(axis=1)) if log else samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)
