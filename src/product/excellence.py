"""Evidence-derived product and academic maturity scoring."""

from __future__ import annotations

from typing import Any


EXPECTED_IDS = {
    "product": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
    "academic": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
}


def validate_scoring_contract(contract: dict[str, Any]) -> dict[str, bool]:
    tracks = contract.get("tracks") or {}
    rules = contract.get("rules") or {}
    execution = contract.get("current_execution") or {}
    shape = set(tracks) == set(EXPECTED_IDS)
    unique_and_ordered = shape and all(
        [row.get("id") for row in tracks[name]] == expected
        for name, expected in EXPECTED_IDS.items()
    )
    numeric = shape and all(
        isinstance(row.get("weight"), int)
        and isinstance(row.get("base_points"), int)
        and 0 <= row["base_points"] <= row["weight"]
        and isinstance(row.get("completion_requires"), list)
        and len(row["completion_requires"])
        == len(set(row["completion_requires"]))
        for rows in tracks.values()
        for row in rows
    )
    return {
        "schema_and_identity": (
            contract.get("schema_version") == 1
            and contract.get("contract_id")
            == "gfs-evidence-derived-excellence-scoring-v1"
        ),
        "track_shape_ids_and_order_are_frozen": unique_and_ordered,
        "weights_and_base_points_are_bounded": numeric,
        "each_track_totals_exactly_100": (
            shape
            and all(
                sum(row["weight"] for row in tracks[name]) == 100
                for name in EXPECTED_IDS
            )
        ),
        "rules_are_fail_closed": (
            rules
            == {
                "all_track_weights_equal_100": True,
                "base_points_require_existing_verified_evidence": True,
                "completion_points_require_all_named_gates": True,
                "negative_academic_results_can_complete_when_valid_and_reproduced": True,
                "product_outcome_thresholds_remain_mandatory": True,
                "score_100_requires_every_category_complete": True,
            }
        ),
        "execution_is_zero_and_not_overridden": (
            execution
            == {
                "scores_are_not_manually_overridden": True,
                "matches_executed": 0,
                "training_executed": False,
                "provider_calls_made": False,
            }
        ),
    }


def derive_excellence(
    contract: dict[str, Any],
    passed_gates: dict[str, bool],
    *,
    baseline_scores: dict[str, int] | None = None,
) -> dict[str, Any]:
    checks = validate_scoring_contract(contract)
    if not all(checks.values()):
        raise ValueError("invalid excellence scoring contract")
    baseline_scores = baseline_scores or {}
    tracks: dict[str, Any] = {}
    for name, rows in contract["tracks"].items():
        categories = []
        for row in rows:
            requirements = row["completion_requires"]
            completed = not requirements or all(
                passed_gates.get(gate) is True for gate in requirements
            )
            points = row["weight"] if completed else row["base_points"]
            categories.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "weight": row["weight"],
                    "points": points,
                    "status": "verified" if completed else "partial",
                    "completion_requires": requirements,
                    "missing_gates": [
                        gate for gate in requirements
                        if passed_gates.get(gate) is not True
                    ],
                }
            )
        score = sum(row["points"] for row in categories)
        maximum = sum(row["weight"] for row in categories)
        tracks[name] = {
            "score": score,
            "baseline_score": int(baseline_scores.get(name, 0)),
            "maximum": maximum,
            "level": "evidence_backed_full_maturity" if score == maximum else "in_progress",
            "categories": categories,
            "critical_gates_remaining": [
                row["id"] for row in categories if row["status"] != "verified"
            ],
        }
    return {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "tracks": tracks,
        "all_tracks_full_maturity": all(
            row["score"] == row["maximum"] for row in tracks.values()
        ),
        "contract_checks": checks,
    }
