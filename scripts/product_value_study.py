"""Audit the frozen comparative product-value protocol or analyze its records."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure import file_sha256  # noqa: E402


PROTOCOL_PATH = ROOT / "data/evaluation/product_value_validation_protocol_v1.json"
PARTICIPANT_ID = re.compile(r"^participant-[a-z0-9]{8,40}$")
MODERATOR_ID = re.compile(r"^moderator-[a-z0-9]{8,40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
OBSERVER_ATTESTATION = (
    "I attest that this record reflects the observed session and contains no "
    "direct participant identifiers or raw free text."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def validate_protocol(protocol: dict[str, Any]) -> dict[str, bool]:
    design = protocol.get("design") or {}
    population = protocol.get("population") or {}
    primary = protocol.get("primary_endpoint") or {}
    safety = protocol.get("safety_and_quality") or {}
    integrity = protocol.get("integrity") or {}
    outputs = protocol.get("outputs") or {}
    current = protocol.get("current_execution") or {}
    return {
        "schema_and_state_are_frozen": (
            protocol.get("schema_version") == 1
            and protocol.get("protocol_id")
            == "gfs-studio-comparative-value-validation-v1"
            and protocol.get("state") == "preregistered_not_executed"
        ),
        "counterbalanced_crossover_is_exact": (
            design.get("type") == "randomized_within_participant_counterbalanced_crossover"
            and design.get("conditions") == ["gfs_studio", "manual_baseline"]
            and design.get("sequences") == ["AB", "BA"]
            and design.get("case_packs") == ["case_pack_alpha", "case_pack_beta"]
            and design.get("maximum_seconds_per_condition") == 900
            and design.get("incorrect_or_incomplete_penalty_seconds") == 900
            and design.get("training_task_before_measurement") is True
            and design.get("same_case_pack_in_both_conditions_forbidden") is True
            and design.get("optional_stopping") is False
        ),
        "sample_and_role_coverage_are_fixed": (
            population.get("target_roles")
            == ["football_analyst", "research_engineer", "product_operator"]
            and population.get("minimum_valid_participants") == 24
            and population.get("minimum_per_role") == 8
            and population.get("minimum_per_sequence") == 12
            and population.get("performance_based_exclusion") is False
            and set(population.get("predeclared_exclusions") or [])
            == {"consent_withdrawn", "infrastructure_failure_before_first_measured_condition"}
        ),
        "primary_endpoint_is_strict_and_reproducible": (
            primary.get("name")
            == "paired_penalized_time_geometric_mean_ratio_gfs_over_baseline"
            and primary.get("threshold") == 0.8
            and primary.get("decision_rule")
            == "upper_bound_of_deterministic_participant_bootstrap_95_percent_ci_must_be_lte_threshold"
            and primary.get("bootstrap_draws") == 10000
            and primary.get("bootstrap_seed") == 20260810
        ),
        "accuracy_and_critical_error_guards_are_fixed": (
            safety.get("accuracy_noninferiority_margin") == -0.05
            and safety.get("accuracy_rule")
            == "lower_bound_of_deterministic_paired_bootstrap_95_percent_ci_for_gfs_minus_baseline_must_be_gte_margin"
            and safety.get("gfs_critical_error_threshold") == 0
            and safety.get("all_valid_records_analyzed") is True
        ),
        "privacy_and_analysis_integrity_are_explicit": (
            integrity.get("protocol_changes_after_first_participant") is False
            and integrity.get("participant_ids_are_pseudonymous") is True
            and integrity.get("moderator_ids_are_pseudonymous") is True
            and integrity.get("raw_free_text_forbidden") is True
            and integrity.get("unknown_record_fields_forbidden") is True
            and integrity.get("completed_conditions_require_sha256_evidence") is True
            and integrity.get("duplicate_participant_ids_forbidden") is True
            and integrity.get("negative_results_are_valid_outputs") is True
            and integrity.get("interim_release_decisions") is False
            and integrity.get("provider_calls_authorized") is False
            and integrity.get("training_authorized") is False
        ),
        "outputs_are_confined_and_exact": (
            set(outputs) == {"raw_records", "decision"}
            and all(
                isinstance(value, str)
                and value.startswith("data/evaluation/product_value_validation_v1/")
                and ".." not in Path(value).parts
                for value in outputs.values()
            )
        ),
        "execution_remains_zero": (
            current.get("participants_observed") == 0
            and current.get("sessions_executed") == 0
            and current.get("matches_executed") == 0
            and current.get("training_executed") is False
            and current.get("provider_calls_made") is False
            and current.get("results_available") is False
        ),
    }


def protocol_report(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol)
    return {
        "schema_version": 1,
        "verification": "gfs_product_value_preregistration",
        "generated_at": _now(),
        "status": "passed_preregistered_not_executed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "study_executed": False,
        "participants_observed": 0,
        "matches_executed": 0,
        "checks": checks,
        "artifact_sha256": {
            "data/evaluation/product_value_validation_protocol_v1.json": file_sha256(protocol_path),
            "scripts/product_value_study.py": file_sha256(Path(__file__)),
            "docs/PRODUCT_VALUE_STUDY.md": file_sha256(ROOT / "docs/PRODUCT_VALUE_STUDY.md"),
        },
        "external_calls_made": False,
        "training_executed": False,
        "provider_calls_made": False,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"invalid JSON on record line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"record line {line_number} is not an object")
        records.append(value)
    if not records:
        raise ValueError("participant record file is empty")
    return records


def _condition_plan(sequence: str) -> list[tuple[str, str]]:
    if sequence == "AB":
        return [("gfs_studio", "case_pack_alpha"), ("manual_baseline", "case_pack_beta")]
    if sequence == "BA":
        return [("manual_baseline", "case_pack_alpha"), ("gfs_studio", "case_pack_beta")]
    raise ValueError("sequence must be AB or BA")


def _validate_record(record: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version", "protocol_id", "participant_id", "target_role",
        "consent", "moderator_id", "sequence", "session_started_at",
        "session_ended_at", "conditions", "excluded", "exclusion_reason",
        "observer_attestation",
    }
    if set(record) != expected:
        raise ValueError("participant record fields must match the privacy allowlist")
    if record.get("schema_version") != 1 or record.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("participant schema or protocol identity mismatch")
    participant_id = str(record.get("participant_id") or "")
    if PARTICIPANT_ID.fullmatch(participant_id) is None:
        raise ValueError("participant_id must be pseudonymous")
    if MODERATOR_ID.fullmatch(str(record.get("moderator_id") or "")) is None:
        raise ValueError(f"moderator_id must be pseudonymous: {participant_id}")
    if record.get("observer_attestation") != OBSERVER_ATTESTATION:
        raise ValueError(f"missing observer attestation: {participant_id}")
    started = _timestamp(record.get("session_started_at"))
    ended = _timestamp(record.get("session_ended_at"))
    if ended <= started:
        raise ValueError(f"non-positive session duration: {participant_id}")
    excluded = record.get("excluded")
    if not isinstance(excluded, bool) or not isinstance(record.get("consent"), bool):
        raise ValueError(f"consent/excluded fields must be boolean: {participant_id}")
    reason = record.get("exclusion_reason")
    if excluded:
        if reason not in protocol["population"]["predeclared_exclusions"]:
            raise ValueError(f"invalid exclusion reason: {participant_id}")
        if record.get("conditions") != []:
            raise ValueError(f"excluded record contains behavioral data: {participant_id}")
        return {"participant_id": participant_id, "excluded": True, "reason": reason}
    if reason is not None or record.get("consent") is not True:
        raise ValueError(f"valid participant lacks consent or has exclusion: {participant_id}")
    role = record.get("target_role")
    if role not in protocol["population"]["target_roles"]:
        raise ValueError(f"invalid target role: {participant_id}")
    sequence = str(record.get("sequence") or "")
    plan = _condition_plan(sequence)
    conditions = record.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 2:
        raise ValueError(f"exactly two conditions are required: {participant_id}")
    normalized: dict[str, dict[str, Any]] = {}
    maximum = float(protocol["design"]["maximum_seconds_per_condition"])
    for row, (expected_condition, expected_case) in zip(conditions, plan):
        if not isinstance(row, dict) or set(row) != {
            "condition", "case_pack", "completed", "correct", "critical_error",
            "duration_seconds", "evidence_sha256",
        }:
            raise ValueError(f"condition fields violate privacy allowlist: {participant_id}")
        if row.get("condition") != expected_condition or row.get("case_pack") != expected_case:
            raise ValueError(f"condition order/case assignment mismatch: {participant_id}")
        if any(not isinstance(row.get(key), bool) for key in ("completed", "correct", "critical_error")):
            raise ValueError(f"condition booleans invalid: {participant_id}")
        if row["correct"] and not row["completed"]:
            raise ValueError(f"incomplete condition cannot be correct: {participant_id}")
        duration = row.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or not 0 < float(duration) <= maximum:
            raise ValueError(f"condition duration invalid: {participant_id}")
        evidence = row.get("evidence_sha256")
        if row["completed"] and SHA256.fullmatch(str(evidence or "")) is None:
            raise ValueError(f"completed condition lacks evidence hash: {participant_id}")
        if not row["completed"] and evidence is not None:
            raise ValueError(f"incomplete condition has evidence hash: {participant_id}")
        normalized[expected_condition] = {
            "case_pack": expected_case,
            "completed": row["completed"],
            "correct": row["correct"],
            "critical_error": row["critical_error"],
            "duration_seconds": float(duration),
        }
    if sum(row["duration_seconds"] for row in normalized.values()) > (ended - started).total_seconds():
        raise ValueError(f"condition durations exceed session window: {participant_id}")
    penalty = float(protocol["design"]["incorrect_or_incomplete_penalty_seconds"])
    for row in normalized.values():
        row["penalized_seconds"] = row["duration_seconds"] if row["completed"] and row["correct"] else penalty
    return {
        "participant_id": participant_id,
        "excluded": False,
        "role": role,
        "sequence": sequence,
        "conditions": normalized,
    }


def _bootstrap_ci(
    values: list[float], draws: int, seed: int, statistic: Callable[[list[float]], float],
) -> list[float]:
    rng = random.Random(seed)
    estimates = []
    count = len(values)
    for _ in range(draws):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        estimates.append(statistic(sample))
    estimates.sort()
    return [estimates[int(draws * 0.025)], estimates[int(draws * 0.975) - 1]]


def _geometric_mean(values: list[float]) -> float:
    return math.exp(mean(math.log(value) for value in values))


def analyze(records_path: Path, protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if not all(validate_protocol(protocol).values()):
        raise ValueError("product value protocol is invalid or has drifted")
    normalized = [_validate_record(row, protocol) for row in _load_records(records_path)]
    ids = [row["participant_id"] for row in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate participant_id")
    valid = [row for row in normalized if not row["excluded"]]
    excluded = [row for row in normalized if row["excluded"]]
    if not valid:
        raise ValueError("no valid participant records")
    roles = Counter(row["role"] for row in valid)
    sequences = Counter(row["sequence"] for row in valid)
    ratios = [
        row["conditions"]["gfs_studio"]["penalized_seconds"]
        / row["conditions"]["manual_baseline"]["penalized_seconds"]
        for row in valid
    ]
    accuracy_differences = [
        float(row["conditions"]["gfs_studio"]["correct"])
        - float(row["conditions"]["manual_baseline"]["correct"])
        for row in valid
    ]
    primary = protocol["primary_endpoint"]
    draws, seed = int(primary["bootstrap_draws"]), int(primary["bootstrap_seed"])
    ratio_ci = _bootstrap_ci(ratios, draws, seed, _geometric_mean)
    accuracy_ci = _bootstrap_ci(accuracy_differences, draws, seed + 1, mean)
    ratio = _geometric_mean(ratios)
    accuracy_difference = mean(accuracy_differences)
    gfs_critical = sum(row["conditions"]["gfs_studio"]["critical_error"] for row in valid)
    minimum_sample_met = len(valid) >= protocol["population"]["minimum_valid_participants"]
    gates = {
        "minimum_valid_sample": minimum_sample_met,
        "minimum_role_coverage": all(roles[role] >= protocol["population"]["minimum_per_role"] for role in protocol["population"]["target_roles"]),
        "counterbalanced_sequence_coverage": all(sequences[value] >= protocol["population"]["minimum_per_sequence"] for value in protocol["design"]["sequences"]),
        "primary_value_threshold": ratio_ci[1] <= float(primary["threshold"]),
        "accuracy_is_noninferior": accuracy_ci[0] >= float(protocol["safety_and_quality"]["accuracy_noninferiority_margin"]),
        "zero_gfs_critical_errors": gfs_critical <= int(protocol["safety_and_quality"]["gfs_critical_error_threshold"]),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "generated_at": _now(),
        "status": "passed_product_value_validation" if passed else "failed_product_value_validation",
        "passed": passed,
        "minimum_sample_met": minimum_sample_met and gates["minimum_role_coverage"] and gates["counterbalanced_sequence_coverage"],
        "primary_value_threshold_met": gates["primary_value_threshold"] and gates["accuracy_is_noninferior"] and gates["zero_gfs_critical_errors"],
        "participants_recorded": len(normalized),
        "participants_valid": len(valid),
        "participants_excluded": len(excluded),
        "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
        "role_counts": dict(sorted(roles.items())),
        "sequence_counts": dict(sorted(sequences.items())),
        "metrics": {
            "paired_penalized_time_geometric_mean_ratio_gfs_over_baseline": ratio,
            "paired_penalized_time_ratio_bootstrap_95": ratio_ci,
            "paired_accuracy_difference_gfs_minus_baseline": accuracy_difference,
            "paired_accuracy_difference_bootstrap_95": accuracy_ci,
            "median_individual_penalized_time_ratio": median(ratios),
            "gfs_critical_errors": gfs_critical,
        },
        "gates": gates,
        "artifact_sha256": {
            "protocol": file_sha256(protocol_path),
            "participant_records": file_sha256(records_path),
            "analyzer": file_sha256(Path(__file__)),
        },
        "participants_observed_by_analyzer": 0,
        "matches_executed_by_analyzer": 0,
        "training_executed_by_analyzer": False,
        "provider_calls_made_by_analyzer": False,
        "limitations": [
            "The result generalizes only to the frozen cases, target roles, and recruited sample.",
            "Pseudonymous observer attestations and evidence hashes do not independently prove participant identity.",
            "A negative result is retained and must not be reinterpreted as product-value evidence.",
        ],
    }


def _atomic_write(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.analyze:
        if args.records is not None:
            parser.error("--records requires --analyze")
        report = protocol_report()
        if args.out:
            _atomic_write(args.out, report, overwrite=args.overwrite)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    if args.records is None or args.out is None:
        parser.error("--analyze requires --records and --out")
    decision = analyze(args.records)
    _atomic_write(args.out, decision, overwrite=args.overwrite)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
