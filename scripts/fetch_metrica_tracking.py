#!/usr/bin/env python3
"""Fetch Metrica sample data and rebuild the compact receiver-ranking artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="", help="Existing metrica-sports/sample-data checkout")
    args = parser.parse_args()
    temporary = None
    if args.source:
        source = Path(args.source).resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="gfs-metrica-")
        source = Path(temporary.name) / "sample-data"
        _run(["git", "clone", "--depth", "1", "https://github.com/metrica-sports/sample-data.git", str(source)])
    out = ROOT / "data/external/metrica"
    for game in (1, 2):
        _run([
            sys.executable, "scripts/build_metrica_pass_dataset.py",
            "--source", str(source), "--game", str(game),
            "--out", str(out / f"pass_candidates_game{game}.csv"),
        ])
    _run([
        sys.executable, "scripts/evaluate_receiver_ranker.py", "--data",
        str(out / "pass_candidates_game1.csv"), str(out / "pass_candidates_game2.csv"),
    ])
    if temporary is not None:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
