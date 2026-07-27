"""Validation and publication transforms for generated simulation content."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any


_FOURTH_WALL = re.compile(
    r"(?i)\b(?:simulation|poisson|rng|glitch|engine\s+bug|broken\s+game)\b"
)
_SCORELINE = re.compile(r"\b\d{1,2}\s*[-:]\s*\d{1,2}\b")


@dataclass
class GenerationAudit:
    kind: str
    accepted: bool
    violations: list[str] = field(default_factory=list)
    original: dict[str, Any] = field(default_factory=dict)
    published: dict[str, Any] = field(default_factory=dict)


def _number(value: Any, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


class GenerationPipeline:
    """Preserve raw generations while producing a bounded publishable payload."""

    def validate_atmosphere(self, payload: Any) -> GenerationAudit:
        original = copy.deepcopy(payload) if isinstance(payload, dict) else {"raw": payload}
        data = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        violations: list[str] = []
        text = str(data.get("key_event", "")).strip()
        if not text:
            text = "The emotional balance around the match begins to shift."
            violations.append("missing_key_event")
        if _SCORELINE.search(text):
            violations.append("invented_scoreline")
            text = _SCORELINE.sub("the scoreline", text)
        if _FOURTH_WALL.search(text):
            violations.append("fourth_wall")
            text = _FOURTH_WALL.sub("match", text)

        raw_signals = data.get("signals") if isinstance(data.get("signals"), dict) else {}
        signals = {
            "coach_pressure": _number(raw_signals.get("coach_pressure"), 0.0, -1.0, 1.0),
            "crowd_hostility": _number(raw_signals.get("crowd_hostility"), 0.0, -1.0, 1.0),
            "player_anxiety": _number(raw_signals.get("player_anxiety"), 0.0, -1.0, 1.0),
            "media_amplification": _number(raw_signals.get("media_amplification"), 0.0, 0.0, 1.0),
            "unity_signal": _number(raw_signals.get("unity_signal"), 0.0, -1.0, 1.0),
        }
        published = {
            "key_event": text[:500],
            "signals": signals,
            "targets": [str(v) for v in data.get("targets", []) if str(v).strip()][:8],
            "persistence": _number(data.get("persistence"), 0.35, 0.0, 1.0),
        }
        return GenerationAudit("atmosphere", not violations, violations, original, published)

    def validate_media(self, payload: Any, facts_ledger: dict[str, Any] | None) -> GenerationAudit:
        original = copy.deepcopy(payload) if isinstance(payload, dict) else {"raw": payload}
        data = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        violations: list[str] = []
        ledger = facts_ledger or {}
        allowed_scores = {
            str(ledger.get(key))
            for key in ("score", "regulation_score", "aet_score", "penalty_score")
            if ledger.get(key) not in (None, "", "not_played")
        }
        for key in ("mainstream", "social_chaos_post"):
            text = str(data.get(key, "")).strip()
            if _FOURTH_WALL.search(text):
                violations.append(f"{key}:fourth_wall")
                text = _FOURTH_WALL.sub("match", text)
            for scoreline in _SCORELINE.findall(text):
                normalized = re.sub(r"\s*[:]\s*", "-", scoreline)
                normalized = re.sub(r"\s*-\s*", "-", normalized)
                if normalized not in allowed_scores:
                    violations.append(f"{key}:invented_scoreline")
                    text = text.replace(scoreline, "the official scoreline")
            data[key] = re.sub(r"\s+", " ", text).strip()
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        data["metrics"] = {
            "professional_score": _number(metrics.get("professional_score"), 0.0, -2.0, 2.0),
            "social_chaos": _number(metrics.get("social_chaos"), 0.0, -5.0, 5.0),
        }
        data["facts_digest"] = {
            key: ledger.get(key)
            for key in ("score", "winner", "total_goals", "drama_score")
            if key in ledger
        }
        return GenerationAudit("media", not violations, violations, original, data)
