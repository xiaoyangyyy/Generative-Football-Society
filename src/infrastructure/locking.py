"""Small cross-platform process leases backed by operating-system file locks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import BinaryIO


class LeaseUnavailable(RuntimeError):
    """Raised when another process owns a requested lease."""


class FileLease:
    """Hold an exclusive lease until release or process exit.

    The lock file is persistent metadata; ownership comes from the OS lock, not
    from file existence. This makes a crashed process recoverable without
    deleting a guessed PID file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: float = 0.0,
        poll_interval: float = 0.05,
    ) -> None:
        self.path = Path(path).resolve()
        self.timeout = max(0.0, float(timeout))
        self.poll_interval = max(0.01, float(poll_interval))
        self._handle: BinaryIO | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    @staticmethod
    def _try_lock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def acquire(self) -> "FileLease":
        if self.held:
            raise RuntimeError("file lease is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            handle = self.path.open("a+b")
            if self.path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                self._try_lock(handle)
            except OSError as exc:
                handle.close()
                if time.monotonic() >= deadline:
                    raise LeaseUnavailable(f"lease is already owned: {self.path}") from exc
                time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))
                continue
            self._handle = handle
            return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self._unlock(handle)
        finally:
            handle.close()

    @classmethod
    def is_held(cls, path: str | Path) -> bool:
        lease = cls(path)
        try:
            lease.acquire()
        except LeaseUnavailable:
            return True
        else:
            lease.release()
            return False

    def __enter__(self) -> "FileLease":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
