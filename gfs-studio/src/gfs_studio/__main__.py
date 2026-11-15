from __future__ import annotations

import argparse
from collections.abc import Sequence

from gfs_studio.bootstrap.application import StudioApplication


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gfs-studio")
    parser.add_argument("--base-dir", required=True)
    args, remaining = parser.parse_known_args(arguments)
    print(StudioApplication(args.base_dir).run_cli(remaining))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
