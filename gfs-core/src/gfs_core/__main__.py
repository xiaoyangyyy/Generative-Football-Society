from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from gfs_core.bootstrap.application import CoreApplication


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gfs-core")
    parser.add_argument("--base-dir")
    args, remaining = parser.parse_known_args(arguments)
    output = CoreApplication(args.base_dir).run_cli(remaining)
    if output:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="backslashreplace")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
