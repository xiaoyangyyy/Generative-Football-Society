"""Budgeted, replayable observations for hidden simulated player quality."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping


SCOUTING_SCHEMA_VERSION = 1
REPORTS_PER_WINDOW = 2
_PLAYER_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def scouting_identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def scouting_observation(
    *, market_id: str, team: str, player_id: str, true_quality: float,
    level: str,
) -> dict[str, Any]:
    if level not in {"baseline", "scouted"}:
        raise ValueError("unsupported scouting observation level")
    if (
        not str(market_id).startswith("free-market-") or not str(team).strip()
        or _PLAYER_ID.fullmatch(str(player_id or "")) is None
        or isinstance(true_quality, bool)
        or not isinstance(true_quality, (int, float))
        or not math.isfinite(float(true_quality))
        or not 0.15 <= float(true_quality) <= 0.92
    ):
        raise ValueError("invalid scouting observation source")
    half_width = 0.075 if level == "baseline" else 0.025
    noise_amplitude = 0.035 if level == "baseline" else 0.012
    noise = (2.0 * _unit("scouting", level, market_id, team, player_id) - 1.0) * noise_amplitude
    estimate = min(0.92, max(0.15, float(true_quality) + noise))
    low = min(float(true_quality), max(0.15, estimate - half_width))
    high = max(float(true_quality), min(0.92, estimate + half_width))
    return {
        "schema_version": 1, "level": level,
        "estimated_quality": round(estimate, 6),
        "quality_low": round(low, 6), "quality_high": round(high, 6),
        "interval_width": round(high - low, 6),
        "coverage_contract": "contains_simulated_truth_by_construction",
        "claim_boundary": (
            "deterministic fictional observation interval; not real scouting, "
            "potential, valuation, or performance evidence"
        ),
    }


def validate_scouting_registry(registry: Mapping[str, Any] | None) -> dict[str, int]:
    if registry is None:
        return {"schema_version": 1, "report_count": 0, "window_count": 0}
    if not isinstance(registry, Mapping) or registry.get("schema_version", 1) != 1:
        raise ValueError("invalid scouting registry")
    reports = registry.get("reports") or []
    if not isinstance(reports, list) or len(reports) > 4096:
        raise ValueError("invalid scouting report history")
    seen = set()
    windows: dict[tuple[str, str], int] = {}
    for report in reports:
        observation = report.get("observation") if isinstance(report, Mapping) else None
        key = (
            str(report.get("market_id") or ""), str(report.get("team") or ""),
            str(report.get("player_id") or ""),
        ) if isinstance(report, Mapping) else ("", "", "")
        window = key[:2]
        if (
            not isinstance(report, Mapping) or report.get("schema_version") != 1
            or not key[0].startswith("free-market-") or not key[1].strip()
            or _PLAYER_ID.fullmatch(key[2]) is None or key in seen
            or not isinstance(report.get("candidate_entry_identity"), str)
            or not str(report.get("target_season_id") or "").startswith("season-")
            or re.fullmatch(r"[0-9a-f]{64}", report["candidate_entry_identity"]) is None
            or not isinstance(observation, Mapping)
            or observation.get("level") != "scouted"
            or observation.get("coverage_contract")
            != "contains_simulated_truth_by_construction"
        ):
            raise ValueError("invalid scouting report")
        numeric = [
            observation.get("estimated_quality"), observation.get("quality_low"),
            observation.get("quality_high"), observation.get("interval_width"),
        ]
        if (
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in numeric)
            or not 0.15 <= observation["quality_low"] <= observation["estimated_quality"] <= observation["quality_high"] <= 0.92
            or round(observation["quality_high"] - observation["quality_low"], 6)
            != observation["interval_width"]
        ):
            raise ValueError("invalid scouting observation interval")
        seen.add(key)
        windows[window] = windows.get(window, 0) + 1
        if windows[window] > REPORTS_PER_WINDOW:
            raise ValueError("scouting report budget exceeded")
    return {
        "schema_version": 1, "report_count": len(reports),
        "window_count": len(windows),
    }


def append_scouting_report(
    registry: Mapping[str, Any] | None, *, market_id: str, team: str,
    entry: Mapping[str, Any], true_quality: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(dict(registry or {"schema_version": 1, "reports": []}))
    validate_scouting_registry(output)
    player_id = str(entry.get("player_id") or "")
    existing = [
        report for report in output.get("reports") or []
        if report.get("market_id") == market_id and report.get("team") == team
    ]
    duplicate = next((report for report in existing if report.get("player_id") == player_id), None)
    if duplicate is not None:
        return output, copy.deepcopy(duplicate)
    if len(existing) >= REPORTS_PER_WINDOW:
        raise ValueError(f"scouting report budget exhausted ({REPORTS_PER_WINDOW})")
    report = {
        "schema_version": 1, "market_id": market_id, "team": team,
        "target_season_id": (
            f"season-{market_id.split('-')[2]}"
            if len(market_id.split('-')) > 2 else ""
        ),
        "player_id": player_id,
        "candidate_entry_identity": scouting_identity(entry),
        "observation": scouting_observation(
            market_id=market_id, team=team, player_id=player_id,
            true_quality=true_quality, level="scouted",
        ),
    }
    output.setdefault("reports", []).append(report)
    validate_scouting_registry(output)
    return output, copy.deepcopy(report)


def reports_for_market(
    registry: Mapping[str, Any] | None, *, market_id: str, team: str,
) -> list[dict[str, Any]]:
    validate_scouting_registry(registry)
    return [
        copy.deepcopy(report) for report in (registry or {}).get("reports") or []
        if report.get("market_id") == market_id and report.get("team") == team
    ]


def scouting_view(registry: Mapping[str, Any] | None) -> dict[str, Any]:
    summary = validate_scouting_registry(registry)
    reports = (registry or {}).get("reports") or []
    public = [
        {
            "market_id": report["market_id"], "team": report["team"],
            "target_season_id": report["target_season_id"],
            "player_id": report["player_id"],
            "observation": copy.deepcopy(report["observation"]),
        }
        for report in reports[-24:]
    ]
    return {
        **summary, "reports_per_window": REPORTS_PER_WINDOW,
        "recent_reports": public,
        "claim_boundary": (
            "fictional bounded information mechanic; not real scouting, "
            "potential, valuation, or performance evidence"
        ),
    }


__all__ = [
    "REPORTS_PER_WINDOW", "append_scouting_report", "reports_for_market",
    "scouting_identity", "scouting_observation", "scouting_view",
    "validate_scouting_registry",
]
