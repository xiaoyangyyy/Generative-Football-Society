#!/usr/bin/env python3
"""Analyze narrative_debug.jsonl anomaly patterns."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "outputs" / "narrative_debug.jsonl"


def main() -> int:
    if not PATH.is_file():
        print(f"No debug log at {PATH}", file=sys.stderr)
        return 1
    mr = []
    for line in PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "match_result":
            mr.append(rec)
    flagged = [m for m in mr if m.get("anomalies")]
    clean = [m for m in mr if not m.get("anomalies")]
    print(f"total={len(mr)} flagged={len(flagged)} clean={len(clean)}")
    ac = Counter()
    for m in flagged:
        for a in m["anomalies"]:
            ac[a.split(":")[0]] += 1
    print("by_type:", dict(ac))
    for label, subset in [("flagged", flagged), ("clean", clean)]:
        if not subset:
            continue
        tg = [m["score"][0] + m["score"][1] for m in subset]
        mxg = [m["micro"]["micro_xg"][0] + m["micro"]["micro_xg"][1] for m in subset]
        shots = [m["micro"]["shots"][0] + m["micro"]["shots"][1] for m in subset]
        print(
            f"{label}: avg_goals={sum(tg)/len(tg):.2f} avg_mxg={sum(mxg)/len(mxg):.2f} "
            f"avg_shots={sum(shots)/len(shots):.1f}"
        )
    pp = Counter(
        (m["tactics_home"]["preset"], m["tactics_away"]["preset"]) for m in flagged
    )
    print("top_preset_pairs:", pp.most_common(8))
    lb = [
        m
        for m in flagged
        if m["tactics_home"]["preset"] == "low_block"
        and m["tactics_away"]["preset"] == "low_block"
    ]
    print(f"low_block vs low_block flagged={len(lb)}")
    for m in lb[:10]:
        sh, sa = m["micro"]["shots"]
        gh, ga = m["score"]
        mxh, mxa = m["micro"]["micro_xg"]
        gph, gpa = m["micro"]["goals_physics"]
        print(
            f"  {m['stage']} | {m['home']} {gh}-{ga} {m['away']} "
            f"shots={sh}-{sa} mxg={mxh:.2f}-{mxa:.2f} gphys={gph}-{gpa}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
