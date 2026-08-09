"""Frozen, prospective live-provider pilot for the cognitive product mode."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.product.workspace import ProductWorkspace, _atomic_json


PILOT_SCHEMA_VERSION = 1
DEFAULT_FIXTURES = (("Brazil", "Argentina"), ("France", "Germany"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PilotProtocol:
    fixtures: tuple[tuple[str, str], ...] = DEFAULT_FIXTURES
    fast: bool = True
    require_provider_evidence: bool = True

    def as_dict(self, *, seed: int) -> dict[str, Any]:
        return {
            "schema_version": PILOT_SCHEMA_VERSION,
            "name": "GFS cognitive live-provider prospective pilot v1",
            "frozen_before_execution": True,
            "fixtures": [
                {"home": home, "away": away, "seed": seed + index}
                for index, (home, away) in enumerate(self.fixtures)
            ],
            "fast": self.fast,
            "call_budget": {
                "coach": 2, "referee": 0, "player": 0,
                "assistant": 0, "crowd": 0,
            },
            "acceptance": {
                "all_matches_complete": True,
                "successful_provider_calls_per_match": "> 0",
                "rule_fallback_only_is_rejected": True,
            },
        }


class ProspectivePilot:
    """Preflight and execute a small cost-bounded provider-backed pilot."""

    def __init__(self, workspace: ProductWorkspace, protocol: PilotProtocol | None = None) -> None:
        self.workspace = workspace
        self.protocol = protocol or PilotProtocol()
        self.report_path = workspace.root / "data/evaluation/llm_prospective_pilot_v1.json"

    def preflight(self) -> dict[str, Any]:
        readiness = self.workspace.readiness()
        blockers = list(readiness["blockers"])
        if self.workspace.config.mode != "cognitive":
            blockers.insert(0, "studio_mode_must_be_cognitive")
        for required in (
            "research_checkpoint_available",
            "research_checkpoint_accepted",
            "llm_credentials_available",
            "llm_config_valid",
        ):
            if not readiness["checks"][required] and required not in blockers:
                blockers.append(required)
        matches_played = len(self.workspace._session().get("matches") or [])
        if matches_played:
            blockers.append("pilot_requires_empty_session")
        return {
            "schema_version": PILOT_SCHEMA_VERSION,
            "state": "ready" if not blockers else "blocked",
            "external_calls_made": False,
            "protocol": self.protocol.as_dict(seed=self.workspace.config.seed),
            "readiness": readiness,
            "provider": readiness.get("llm_provider"),
            "matches_played": matches_played,
            "blockers": blockers,
        }

    def execute(self) -> dict[str, Any]:
        preflight = self.preflight()
        if preflight["blockers"]:
            raise RuntimeError("pilot is not ready: " + ", ".join(preflight["blockers"]))

        started_at = _now()
        matches: list[dict[str, Any]] = []
        failure = ""
        for home, away in self.protocol.fixtures:
            report = self.workspace.run_match(home, away, fast=self.protocol.fast)
            provider = report["layers"]["cognition"]["provider"]
            accepted = bool(provider.get("real_provider_evidence"))
            matches.append({
                "match_id": report["match_id"],
                "fixture": report["fixture"],
                "provider": provider,
                "accepted": accepted,
                "report": Path(report["report_path"]).relative_to(self.workspace.root).as_posix(),
                "dashboard": Path(report["dashboard_path"]).relative_to(self.workspace.root).as_posix(),
                "cognitive_log": report["artifacts"].get("cognitive_log"),
            })
            if self.protocol.require_provider_evidence and not accepted:
                failure = f"no successful provider call for {home} vs {away}"
                break

        passed = len(matches) == len(self.protocol.fixtures) and all(
            item["accepted"] for item in matches
        )
        result = {
            "schema_version": PILOT_SCHEMA_VERSION,
            "state": "passed" if passed else "failed",
            "started_at": started_at,
            "completed_at": _now(),
            "protocol": preflight["protocol"],
            "external_calls_made": True,
            "matches": matches,
            "successful_provider_calls": sum(
                int(item["provider"].get("successful_calls", 0)) for item in matches
            ),
            "failure": failure or None,
            "promotion_authorized": False,
            "note": "Pilot evidence is necessary but does not by itself authorize production promotion.",
        }
        _atomic_json(self.report_path, result)
        if not passed:
            raise RuntimeError(f"prospective pilot rejected: {failure or 'incomplete run'}")
        return result
