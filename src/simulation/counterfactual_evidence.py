"""Persistence contract for matched-seed tactical counterfactual reports."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


COUNTERFACTUAL_EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_COUNTERFACTUAL_EVIDENCE_PATH = Path(
    "data/persistence/tactical_counterfactuals.jsonl"
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def counterfactual_evidence_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / DEFAULT_COUNTERFACTUAL_EVIDENCE_PATH


def _evaluation_id(report: dict[str, Any]) -> str:
    identity = {
        key: report.get(key)
        for key in (
            "team",
            "opponent",
            "baseline_preset",
            "treatment_preset",
            "root_seed",
            "samples",
            "match_seconds",
            "fast_mode",
        )
    }
    encoded = json.dumps(
        identity, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.blake2s(encoded, digest_size=12).hexdigest()


def append_counterfactual_evidence(
    base_dir: str | Path,
    report: dict[str, Any],
) -> Path:
    """Persist a compact matched-seed result for later trust guidance."""
    path = counterfactual_evidence_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    interval = list(report.get("confidence_interval") or [0.0, 0.0])
    interval = (interval + [0.0, 0.0])[:2]
    record = {
        "schema_version": COUNTERFACTUAL_EVIDENCE_SCHEMA_VERSION,
        "evaluation_id": _evaluation_id(report),
        "team": str(report.get("team", "")),
        "opponent": str(report.get("opponent", "")),
        "baseline_preset": str(report.get("baseline_preset", "balanced")),
        "treatment_preset": str(report.get("treatment_preset", "")),
        "samples": int(max(0, _finite(report.get("samples")))),
        "average_treatment_effect": _finite(
            report.get("average_treatment_effect")
        ),
        "confidence_interval": [_finite(value) for value in interval],
        "directionally_supported": bool(
            report.get("directionally_supported", False)
        ),
        "match_seconds": _finite(report.get("match_seconds")),
        "fast_mode": bool(report.get("fast_mode", False)),
        "causal_interpretation": "within_simulator_only",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def load_counterfactual_evidence(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid counterfactual evidence JSON at line {line_number}"
                ) from exc
            if int(record.get("schema_version", 0)) != (
                COUNTERFACTUAL_EVIDENCE_SCHEMA_VERSION
            ):
                raise ValueError(
                    f"unsupported counterfactual evidence schema at line {line_number}"
                )
            evaluation_id = str(record.get("evaluation_id", ""))
            if not evaluation_id:
                raise ValueError(
                    f"missing counterfactual evaluation_id at line {line_number}"
                )
            by_id[evaluation_id] = record
    return list(by_id.values())
