from __future__ import annotations

import os
import time
from pathlib import Path
from typing import BinaryIO


class LeaseUnavailable(RuntimeError):
    pass


class FileLease:
    def __init__(self, path: str | Path, *, timeout: float = 0.0) -> None:
        self.path = Path(path).resolve()
        self.timeout = max(0.0, float(timeout))
        self._handle: BinaryIO | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> FileLease:
        if self.held:
            raise RuntimeError("file lease is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            handle = self.path.open("a+b")
            if self.path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                _lock(handle)
            except OSError as error:
                handle.close()
                if time.monotonic() >= deadline:
                    raise LeaseUnavailable(str(self.path)) from error
                time.sleep(0.05)
                continue
            self._handle = handle
            return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            _unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> FileLease:
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


def _lock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
