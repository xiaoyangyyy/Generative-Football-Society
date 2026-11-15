from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


class RosterRepository:
    def __init__(self, project_root: str | Path) -> None:
        self._root = Path(project_root) / "data" / "rosters"

    def load(
        self, team: str, override: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if override is not None:
            return _sanitize(dict(override))
        safe_name = re.sub(r"[^\w\-]+", "_", team)
        path = self._root / f"{safe_name}.json"
        if not path.is_file():
            return None
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("roster document is oversized")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("roster document must be a JSON object")
        return _sanitize(payload)

    @staticmethod
    def identity(roster: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(roster),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _sanitize(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            result = _sanitize(item)
            if result is not _MISSING:
                cleaned[str(key)] = result
        return cleaned
    if isinstance(value, list):
        return [
            None if result is _MISSING else result
            for item in value
            for result in [_sanitize(item)]
        ]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return value if math.isfinite(float(value)) else _MISSING


_MISSING = object()
