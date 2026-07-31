"""Evaluate outcome-linked world-model/LLM tactical fusion decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.fusion_audit import (  # noqa: E402
    DEFAULT_FUSION_AUDIT_PATH,
    evaluate_fusion_audits,
    load_fusion_audits,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate world-model/LLM tactical fusion audit records.",
    )
    parser.add_argument(
        "--audit-log",
        default=str(ROOT / DEFAULT_FUSION_AUDIT_PATH),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports/evaluation/world_model_llm_fusion.json"),
    )
    parser.add_argument("--min-records", type=int, default=20)
    args = parser.parse_args()

    records = load_fusion_audits(args.audit_log)
    if not records:
        print(
            "No fusion audit records found. Run matches with MATCH_WORLD_MODEL=1 "
            "and MATCH_WM_PLAN=1 first.",
            file=sys.stderr,
        )
        return 2
    report = evaluate_fusion_audits(records, min_records=args.min_records)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["ready_for_observational_comparison"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
