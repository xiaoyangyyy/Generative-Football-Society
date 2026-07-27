#!/usr/bin/env python3
"""Thin root entrypoint for the unified GFS CLI."""

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
