#!/usr/bin/env python3
"""Run roster ability cross-check vs Transfermarkt."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data_engine.roster_verification import write_verification_report


def main() -> None:
    out = write_verification_report(ROOT)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
