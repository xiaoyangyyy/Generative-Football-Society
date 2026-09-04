from pathlib import Path
from types import SimpleNamespace

import pytest

from src.product.pilot import ProspectivePilot


class _PilotWorkspace:
    def __init__(self, root: Path, reports: list[dict]):
        self.root = root
        self.config = SimpleNamespace(mode="cognitive", seed=17)
        self._reports = iter(reports)

    @staticmethod
    def readiness():
        return {
            "blockers": [],
            "checks": {
                "research_checkpoint_available": True,
                "research_checkpoint_accepted": True,
                "llm_credentials_available": True,
                "llm_config_valid": True,
            },
            "llm_provider": {"provider": "openai_compatible"},
        }

    @staticmethod
    def _session():
        return {"matches": []}

    def run_match(self, *_args, **_kwargs):
        return next(self._reports)


def _report(root: Path, index: int, *, accepted: bool = True) -> dict:
    calls = 1
    transport_calls = calls if accepted else 0
    return {
        "match_id": f"pilot-match-{index}",
        "fixture": {"home": "Home", "away": "Away", "seed": index},
        "report_path": str(root / f"report-{index}.json"),
        "dashboard_path": str(root / f"report-{index}.html"),
        "artifacts": {"cognitive_log": f"log-{index}.json"},
        "integrity": {"accepted": accepted},
        "layers": {
            "cognition": {
                "provider": {
                    "successful_calls": calls,
                    "real_provider_evidence": True,
                    "transport": {
                        "available": True,
                        "truncated": False,
                        "successful_calls": transport_calls,
                        "contains_prompts_or_credentials": False,
                    },
                },
            },
        },
    }


def test_prospective_pilot_requires_complete_transport_evidence(tmp_path):
    reports = [_report(tmp_path, 1), _report(tmp_path, 2)]
    result = ProspectivePilot(_PilotWorkspace(tmp_path, reports)).execute()

    assert result["state"] == "passed"
    assert result["successful_provider_calls"] == 2
    assert all(item["accepted"] for item in result["matches"])


def test_prospective_pilot_rejects_transport_or_report_integrity_gap(tmp_path):
    reports = [_report(tmp_path, 1, accepted=False)]
    pilot = ProspectivePilot(_PilotWorkspace(tmp_path, reports))

    with pytest.raises(RuntimeError, match="transport integrity rejected"):
        pilot.execute()
    assert pilot.report_path.is_file()
