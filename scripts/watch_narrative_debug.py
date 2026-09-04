#!/usr/bin/env python3
"""Summarize narrative_debug.jsonl anomalies (LLM stack debug)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "outputs" / "narrative_debug.jsonl"


def main() -> int:
    if not PATH.is_file():
        print(f"No debug log at {PATH}")
        return 1
    matches = []
    for line in PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "match_result":
            matches.append(rec)
    print(f"matches_logged={len(matches)}")
    flagged = [m for m in matches if m.get("anomalies")]
    print(f"anomaly_matches={len(flagged)}")
    for m in flagged:
        h, a = m["home"], m["away"]
        sc = m["score"]
        print(
            f"  {m['stage']} | {h} {sc[0]}-{sc[1]} {a} | "
            f"anomalies={m['anomalies']} | "
            f"presets={m['tactics_home'].get('preset')}/{m['tactics_away'].get('preset')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
