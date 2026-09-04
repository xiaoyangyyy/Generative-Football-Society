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
from src.product.human_study import (  # noqa: E402
    ARCHIVE_LAYOUT,
    bind_records_to_registry,
    register_participant,
    validate_registry,
    verify_content_addressed_archive,
)


PROTOCOL_PATH = ROOT / "data/evaluation/product_value_validation_protocol_v1.json"
CASE_PACK_PATH = ROOT / "data/evaluation/product_value_case_packs_v1.json"
DEFAULT_REGISTRY_PATH = (
    ROOT / "data/evaluation/product_value_validation_v1/session_registry.json"
)
DEFAULT_RECORDS_PATH = (
    ROOT / "data/evaluation/product_value_validation_v1/participant_records.jsonl"
)
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
    allocation = protocol.get("allocation") or {}
    registration = protocol.get("registration") or {}
    archive = protocol.get("evidence_archive") or {}
    amendments = protocol.get("preexecution_amendments") or []
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
        "allocation_is_frozen_and_role_stratified": (
            allocation.get("method") == "role_stratified_permuted_blocks_v1"
            and allocation.get("seed") == 20260905
            and allocation.get("block_size") == 4
            and allocation.get("sequences") == ["AB", "BA"]
            and allocation.get("assignments_per_sequence_per_block") == 2
        ),
        "sample_and_role_coverage_are_fixed": (
            population.get("target_roles")
            == ["football_analyst", "research_engineer", "product_operator"]
            and population.get("minimum_valid_participants") == 24
            and population.get("minimum_per_role") == 8
            and population.get("minimum_per_sequence") == 12
            and population.get("minimum_per_role_sequence") == 4
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
            and integrity.get("authoritative_session_registry_required") is True
            and integrity.get("all_registrations_analyzed") is True
            and integrity.get("record_identity_bound_to_registration") is True
            and integrity.get("sequence_assigned_by_frozen_allocation") is True
            and integrity.get("case_pack_identity_bound_at_registration") is True
            and integrity.get("evidence_hashes_verified_against_real_files") is True
            and integrity.get("correctness_recomputed_from_structured_receipts")
            is True
        ),
        "registration_is_authoritative_and_premeasurement": (
            registration.get("registry_kind")
            == "gfs_human_study_session_registry_v1"
            and registration.get("participant_id_pattern")
            == r"^participant-[a-z0-9]{8,40}$"
            and registration.get("moderator_id_pattern")
            == r"^moderator-[a-z0-9]{8,40}$"
            and registration.get("registration_before_measurement") is True
            and registration.get("consent_required") is True
            and registration.get("duplicate_registration_policy")
            == "idempotent_exact_match"
        ),
        "evidence_archive_requires_scored_real_bytes": (
            archive.get("layout") == ARCHIVE_LAYOUT
            and archive.get("file_name_is_exact_sha256") is True
            and archive.get("symlinks_allowed") is False
            and archive.get("all_referenced_content_verified_during_analysis")
            is True
            and archive.get("duplicate_digest_across_observations_allowed")
            is False
            and archive.get("archive_path_recorded_in_decision") is False
            and archive.get("structured_receipts_are_rescored_from_frozen_keys")
            is True
        ),
        "preexecution_amendment_is_zero_participant": (
            len(amendments) == 1
            and amendments[0].get("amendment_id")
            == "authoritative-randomized-value-study-v1"
            and amendments[0].get("participants_observed_before_amendment") == 0
            and amendments[0].get("sessions_executed_before_amendment") == 0
            and amendments[0].get("results_available_before_amendment") is False
        ),
        "outputs_are_confined_and_exact": (
            set(outputs)
            == {"session_registry", "case_pack_manifest", "raw_records", "decision"}
            and outputs.get("case_pack_manifest")
            == "data/evaluation/product_value_case_packs_v1.json"
            and all(
                isinstance(outputs.get(name), str)
                and outputs[name].startswith(
                    "data/evaluation/product_value_validation_v1/"
                )
                and ".." not in Path(outputs[name]).parts
                for name in ("session_registry", "raw_records", "decision")
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


def validate_case_packs(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, bool]:
    packs = manifest.get("packs") or []
    delivery = manifest.get("delivery_contract") or {}
    receipt = manifest.get("receipt_contract") or {}
    balance = manifest.get("balance_contract") or {}
    current = manifest.get("current_execution") or {}
    expected_questions = [
        "recommended_intervention",
        "supported_claim",
        "next_evidence_action",
    ]
    exact_pack_fields = all(
        isinstance(pack, dict)
        and set(pack) == {
            "case_pack_id",
            "revision",
            "participant_packet",
            "scoring_key",
        }
        for pack in packs
    )
    question_contract = exact_pack_fields and all(
        [
            row.get("id")
            for row in (pack.get("participant_packet") or {}).get("questions", [])
        ]
        == expected_questions
        and set(pack.get("scoring_key") or {}) == set(expected_questions)
        and all(
            (pack.get("scoring_key") or {}).get(question["id"])
            in question.get("allowed_answers", [])
            for question in (pack.get("participant_packet") or {}).get(
                "questions", []
            )
        )
        for pack in packs
    )
    derived_winners: list[str] = []
    numeric_contract_valid = exact_pack_fields
    if exact_pack_fields:
        for pack in packs:
            packet = pack.get("participant_packet") or {}
            branches = packet.get("branches")
            rules = packet.get("decision_contract")
            if (
                not isinstance(branches, list)
                or not branches
                or not isinstance(rules, list)
                or not rules
                or len({row.get("id") for row in branches}) != len(branches)
            ):
                numeric_contract_valid = False
                break
            winners = []
            for branch in branches:
                satisfied = True
                for rule in rules:
                    if not isinstance(rule, dict) or set(rule) != {
                        "field",
                        "operator",
                        "threshold",
                    }:
                        satisfied = False
                        numeric_contract_valid = False
                        break
                    value = branch.get(rule.get("field"))
                    threshold = rule.get("threshold")
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or isinstance(threshold, bool)
                        or not isinstance(threshold, (int, float))
                        or not math.isfinite(float(value))
                        or not math.isfinite(float(threshold))
                        or rule.get("operator") not in {"gte", "lte"}
                    ):
                        satisfied = False
                        numeric_contract_valid = False
                        break
                    if rule["operator"] == "gte":
                        satisfied = satisfied and float(value) >= float(threshold)
                    else:
                        satisfied = satisfied and float(value) <= float(threshold)
                if satisfied:
                    winners.append(str(branch.get("id")))
            if len(winners) != 1:
                numeric_contract_valid = False
                break
            derived_winners.append(winners[0])
    return {
        "schema_identity_and_state_are_frozen": (
            manifest.get("schema_version") == 1
            and manifest.get("manifest_id")
            == "gfs-studio-product-value-case-packs-v1"
            and manifest.get("protocol_id") == protocol.get("protocol_id")
            and manifest.get("state") == "frozen_preexecution"
        ),
        "two_named_packs_are_exact": (
            exact_pack_fields
            and [pack.get("case_pack_id") for pack in packs]
            == protocol["design"]["case_packs"]
            and all(pack.get("revision") == 1 for pack in packs)
        ),
        "questions_and_scoring_keys_are_exact": question_contract,
        "recommended_interventions_are_numerically_derived": (
            numeric_contract_valid
            and len(derived_winners) == len(packs)
            and all(
                pack["scoring_key"]["recommended_intervention"] == winner
                for pack, winner in zip(packs, derived_winners)
            )
        ),
        "delivery_prevents_key_and_case_leakage": (
            delivery.get("operator_scoring_keys_hidden_until_condition_submission")
            is True
            and delivery.get("same_pack_in_both_conditions_forbidden") is True
            and delivery.get("training_case_is_separate_and_unscored") is True
            and bool(delivery.get("gfs_studio"))
            and bool(delivery.get("manual_baseline"))
        ),
        "structured_receipt_contract_is_exact": (
            receipt.get("schema_version") == 1
            and receipt.get("maximum_bytes") == 65536
            and receipt.get("required_fields")
            == [
                "schema_version",
                "protocol_id",
                "registration_id",
                "participant_id",
                "condition",
                "case_pack",
                "answers",
                "submitted_at",
            ]
            and receipt.get("answer_fields") == expected_questions
        ),
        "case_balance_is_explicit": all(
            balance.get(key) is True
            for key in (
                "same_question_ids_and_order",
                "same_answer_cardinality",
                "one_branch_satisfies_all_thresholds_per_pack",
                "identical_claim_boundary_answers",
                "pilot_difficulty_matching_required_before_first_measured_session",
            )
        ),
        "execution_remains_zero": (
            current.get("participants_exposed") == 0
            and current.get("measured_sessions") == 0
            and current.get("results_available") is False
        ),
    }


def protocol_report(protocol_path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    checks = validate_protocol(protocol)
    case_pack_checks = validate_case_packs(_read_json(CASE_PACK_PATH), protocol)
    combined_checks = {
        **checks,
        **{f"case_pack_{key}": value for key, value in case_pack_checks.items()},
    }
    return {
        "schema_version": 1,
        "verification": "gfs_product_value_preregistration",
        "generated_at": _now(),
        "status": "passed_preregistered_not_executed"
        if all(combined_checks.values())
        else "failed",
        "passed": all(combined_checks.values()),
        "study_executed": False,
        "participants_observed": 0,
        "matches_executed": 0,
        "checks": combined_checks,
        "artifact_sha256": {
            "data/evaluation/product_value_validation_protocol_v1.json": file_sha256(protocol_path),
            "scripts/product_value_study.py": file_sha256(Path(__file__)),
            "docs/PRODUCT_VALUE_STUDY.md": file_sha256(ROOT / "docs/PRODUCT_VALUE_STUDY.md"),
            "data/evaluation/product_value_case_packs_v1.json": file_sha256(
                CASE_PACK_PATH
            ),
            "src/product/human_study.py": file_sha256(
                ROOT / "src/product/human_study.py"
            ),
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
        "schema_version", "protocol_id", "registration_id", "participant_id",
        "target_role", "consent", "moderator_id", "sequence", "session_started_at",
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
    registration_id = str(record.get("registration_id") or "")
    if re.fullmatch(r"REG-[0-9A-F]{24}", registration_id) is None:
        raise ValueError(f"registration_id is invalid: {participant_id}")
    moderator_id = str(record.get("moderator_id") or "")
    if MODERATOR_ID.fullmatch(moderator_id) is None:
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
    role = record.get("target_role")
    if role not in protocol["population"]["target_roles"]:
        raise ValueError(f"invalid target role: {participant_id}")
    sequence = str(record.get("sequence") or "")
    plan = _condition_plan(sequence)
    reason = record.get("exclusion_reason")
    if excluded:
        if reason not in protocol["population"]["predeclared_exclusions"]:
            raise ValueError(f"invalid exclusion reason: {participant_id}")
        if record.get("conditions") != []:
            raise ValueError(f"excluded record contains behavioral data: {participant_id}")
        return {
            "participant_id": participant_id,
            "registration_id": registration_id,
            "moderator_id": moderator_id,
            "session_started_at": record["session_started_at"],
            "session_ended_at": record["session_ended_at"],
            "excluded": True,
            "reason": reason,
            "role": role,
            "sequence": sequence,
        }
    if reason is not None or record.get("consent") is not True:
        raise ValueError(f"valid participant lacks consent or has exclusion: {participant_id}")
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
            "evidence_sha256": evidence,
        }
    if sum(row["duration_seconds"] for row in normalized.values()) > (ended - started).total_seconds():
        raise ValueError(f"condition durations exceed session window: {participant_id}")
    penalty = float(protocol["design"]["incorrect_or_incomplete_penalty_seconds"])
    for row in normalized.values():
        row["penalized_seconds"] = row["duration_seconds"] if row["completed"] and row["correct"] else penalty
    return {
        "participant_id": participant_id,
        "registration_id": registration_id,
        "moderator_id": moderator_id,
        "session_started_at": record["session_started_at"],
        "session_ended_at": record["session_ended_at"],
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


def _score_receipt(
    content: bytes,
    record: dict[str, Any],
    condition_name: str,
    manifest: dict[str, Any],
) -> bool:
    try:
        receipt = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"receipt is not valid UTF-8 JSON: {record['participant_id']}"
        ) from exc
    if not isinstance(receipt, dict):
        raise ValueError(
            f"receipt must be a JSON object: {record['participant_id']}"
        )
    contract = manifest["receipt_contract"]
    if set(receipt) != set(contract["required_fields"]):
        raise ValueError(
            f"receipt fields violate the frozen contract: {record['participant_id']}"
        )
    condition = record["conditions"][condition_name]
    expected = {
        "schema_version": contract["schema_version"],
        "protocol_id": manifest["protocol_id"],
        "registration_id": record["registration_id"],
        "participant_id": record["participant_id"],
        "condition": condition_name,
        "case_pack": condition["case_pack"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"receipt identity does not match the measured condition: "
            f"{record['participant_id']}"
        )
    submitted = _timestamp(receipt.get("submitted_at"))
    if not (
        _timestamp(record["session_started_at"])
        <= submitted
        <= _timestamp(record["session_ended_at"])
    ):
        raise ValueError(
            f"receipt timestamp is outside the session: {record['participant_id']}"
        )
    answers = receipt.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(contract["answer_fields"]):
        raise ValueError(
            f"receipt answers violate the frozen contract: {record['participant_id']}"
        )
    packs = {pack["case_pack_id"]: pack for pack in manifest["packs"]}
    pack = packs[condition["case_pack"]]
    questions = {
        question["id"]: question
        for question in pack["participant_packet"]["questions"]
    }
    if any(
        answers[question_id] not in questions[question_id]["allowed_answers"]
        for question_id in contract["answer_fields"]
    ):
        raise ValueError(
            f"receipt contains an answer outside the frozen choices: "
            f"{record['participant_id']}"
        )
    return all(
        answers[question_id] == pack["scoring_key"][question_id]
        for question_id in contract["answer_fields"]
    )


def analyze(
    records_path: Path,
    registry_path: Path,
    evidence_root: Path,
    protocol_path: Path = PROTOCOL_PATH,
    case_pack_path: Path = CASE_PACK_PATH,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if not all(validate_protocol(protocol).values()):
        raise ValueError("product value protocol is invalid or has drifted")
    case_manifest = _read_json(case_pack_path)
    case_pack_checks = validate_case_packs(case_manifest, protocol)
    if not all(case_pack_checks.values()):
        raise ValueError("product value case packs are invalid or have drifted")
    normalized = [_validate_record(row, protocol) for row in _load_records(records_path)]
    registry_report = validate_registry(
        protocol_path=protocol_path,
        registry=registry_path,
        case_pack_manifest_path=case_pack_path,
    )
    registry_binding = bind_records_to_registry(normalized, registry_report)
    valid = [row for row in normalized if not row["excluded"]]
    excluded = [row for row in normalized if row["excluded"]]
    if not valid:
        raise ValueError("no valid participant records")
    evidence_digests = [
        str(condition["evidence_sha256"])
        for row in valid
        for condition in row["conditions"].values()
        if condition["evidence_sha256"] is not None
    ]
    archive_report = verify_content_addressed_archive(
        evidence_root,
        evidence_digests,
        retain_verified_bytes=True,
        max_bytes_per_artifact=case_manifest["receipt_contract"]["maximum_bytes"],
    )
    penalty = float(protocol["design"]["incorrect_or_incomplete_penalty_seconds"])
    for row in valid:
        for condition_name, condition in row["conditions"].items():
            if condition["completed"]:
                derived_correct = _score_receipt(
                    archive_report["verified_bytes"][condition["evidence_sha256"]],
                    row,
                    condition_name,
                    case_manifest,
                )
                if condition["correct"] is not derived_correct:
                    raise ValueError(
                        f"declared correctness disagrees with frozen scoring: "
                        f"{row['participant_id']}:{condition_name}"
                    )
                condition["correct"] = derived_correct
            condition["penalized_seconds"] = (
                condition["duration_seconds"]
                if condition["completed"] and condition["correct"]
                else penalty
            )
    roles = Counter(row["role"] for row in valid)
    sequences = Counter(row["sequence"] for row in valid)
    role_sequences = Counter((row["role"], row["sequence"]) for row in valid)
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
        "role_stratified_sequence_coverage": all(
            role_sequences[(role, sequence)]
            >= protocol["population"]["minimum_per_role_sequence"]
            for role in protocol["population"]["target_roles"]
            for sequence in protocol["design"]["sequences"]
        ),
        "primary_value_threshold": ratio_ci[1] <= float(primary["threshold"]),
        "accuracy_is_noninferior": accuracy_ci[0] >= float(protocol["safety_and_quality"]["accuracy_noninferiority_margin"]),
        "zero_gfs_critical_errors": gfs_critical <= int(protocol["safety_and_quality"]["gfs_critical_error_threshold"]),
        "all_records_bound_to_registry": all(
            value is True
            for key, value in registry_binding.items()
            if key.startswith("all_")
        ),
        "all_evidence_bytes_verified_and_rescored": (
            archive_report["verified_artifacts"] == len(evidence_digests)
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "generated_at": _now(),
        "status": "passed_product_value_validation" if passed else "failed_product_value_validation",
        "passed": passed,
        "minimum_sample_met": (
            minimum_sample_met
            and gates["minimum_role_coverage"]
            and gates["counterbalanced_sequence_coverage"]
            and gates["role_stratified_sequence_coverage"]
        ),
        "primary_value_threshold_met": gates["primary_value_threshold"] and gates["accuracy_is_noninferior"] and gates["zero_gfs_critical_errors"],
        "participants_recorded": len(normalized),
        "participants_valid": len(valid),
        "participants_excluded": len(excluded),
        "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
        "role_counts": dict(sorted(roles.items())),
        "sequence_counts": dict(sorted(sequences.items())),
        "role_sequence_counts": {
            f"{role}:{sequence}": role_sequences[(role, sequence)]
            for role in protocol["population"]["target_roles"]
            for sequence in protocol["design"]["sequences"]
        },
        "metrics": {
            "paired_penalized_time_geometric_mean_ratio_gfs_over_baseline": ratio,
            "paired_penalized_time_ratio_bootstrap_95": ratio_ci,
            "paired_accuracy_difference_gfs_minus_baseline": accuracy_difference,
            "paired_accuracy_difference_bootstrap_95": accuracy_ci,
            "median_individual_penalized_time_ratio": median(ratios),
            "gfs_critical_errors": gfs_critical,
        },
        "gates": gates,
        "case_pack_checks": case_pack_checks,
        "registry_checks": registry_binding,
        "evidence_archive": {
            "layout": archive_report["layout"],
            "verified_artifacts": archive_report["verified_artifacts"],
            "inventory_sha256": archive_report["inventory_sha256"],
        },
        "artifact_sha256": {
            "protocol": file_sha256(protocol_path),
            "case_pack_manifest": file_sha256(case_pack_path),
            "session_registry": file_sha256(registry_path),
            "participant_records": file_sha256(records_path),
            "analyzer": file_sha256(Path(__file__)),
        },
        "participants_observed_by_analyzer": 0,
        "matches_executed_by_analyzer": 0,
        "training_executed_by_analyzer": False,
        "provider_calls_made_by_analyzer": False,
        "limitations": [
            "The result generalizes only to the frozen cases, target roles, and recruited sample.",
            "Byte-verified pseudonymous receipts do not independently prove a participant's real-world identity.",
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
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--case-pack-manifest", type=Path, default=CASE_PACK_PATH)
    parser.add_argument("--participant-id")
    parser.add_argument("--target-role")
    parser.add_argument("--moderator-id")
    parser.add_argument("--confirm-consent", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.analyze and args.register:
        parser.error("--analyze and --register are mutually exclusive")
    if args.register:
        if args.out is not None or args.evidence_root is not None or args.overwrite:
            parser.error("--register does not accept analysis output options")
        if not all((args.participant_id, args.target_role, args.moderator_id)):
            parser.error(
                "--register requires --participant-id, --target-role, and --moderator-id"
            )
        if not args.confirm_consent:
            parser.error("--register requires --confirm-consent")
        protocol = _read_json(PROTOCOL_PATH)
        if not all(validate_protocol(protocol).values()):
            raise ValueError("cannot register under an invalid or drifted protocol")
        if not all(
            validate_case_packs(_read_json(args.case_pack_manifest), protocol).values()
        ):
            raise ValueError("cannot register under invalid or drifted case packs")
        result = register_participant(
            project_root=ROOT,
            protocol_path=PROTOCOL_PATH,
            registry_path=args.registry or DEFAULT_REGISTRY_PATH,
            records_path=args.records or DEFAULT_RECORDS_PATH,
            participant_id=args.participant_id,
            target_role=args.target_role,
            moderator_id=args.moderator_id,
            consent_recorded=True,
            case_pack_manifest_path=args.case_pack_manifest,
        )
        public = {
            "schema_version": 1,
            "status": "registered" if result["created"] else "already_registered",
            "created": result["created"],
            "registration": result["registration"],
            "external_calls_made": False,
            "matches_executed": 0,
            "training_executed": False,
        }
        print(json.dumps(public, ensure_ascii=False, indent=2))
        return 0
    if not args.analyze:
        if any(
            value is not None
            for value in (
                args.records,
                args.registry,
                args.evidence_root,
                args.participant_id,
                args.target_role,
                args.moderator_id,
            )
        ) or args.confirm_consent or args.case_pack_manifest != CASE_PACK_PATH:
            parser.error("study inputs require --register or --analyze")
        report = protocol_report()
        if args.out:
            _atomic_write(args.out, report, overwrite=args.overwrite)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    if any((args.participant_id, args.target_role, args.moderator_id)) or args.confirm_consent:
        parser.error("participant registration options require --register")
    if (
        args.records is None
        or args.registry is None
        or args.evidence_root is None
        or args.out is None
    ):
        parser.error(
            "--analyze requires --records, --registry, --evidence-root, and --out"
        )
    decision = analyze(
        args.records,
        args.registry,
        args.evidence_root,
        case_pack_path=args.case_pack_manifest,
    )
    _atomic_write(args.out, decision, overwrite=args.overwrite)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
