"""Filesystem durability helpers for authoritative local state."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_directory(path: str | Path) -> bool:
    """Flush directory metadata where Python exposes a portable descriptor.

    Same-directory replacement protects readers from partial files. On POSIX,
    flushing the containing directory additionally makes the rename durable
    across a host crash. Windows does not expose an equivalent directory
    handle through os.open, so process-crash atomicity there remains provided
    by os.replace and the file flush.
    """

    if os.name == "nt":
        return False
    directory = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True

