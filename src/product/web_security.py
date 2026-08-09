"""Authentication and trusted-proxy boundary for the Studio Web surface."""

from __future__ import annotations

import hashlib
import ipaddress
import math
import re
import secrets
import threading
import time
from collections import deque
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import urlsplit


SESSION_COOKIE = "__Host-gfs_studio_session"
HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
PLACEHOLDER_TOKENS = {
    "", "change-me", "replace-me", "replace_with_random_token",
    "replace_with_32_or_more_random_characters",
    "your_access_token", "your_token_here",
}


def resolve_web_access_token(values: Mapping[str, str]) -> str:
    direct = str(values.get("GFS_WEB_ACCESS_TOKEN") or "").strip()
    secret_file = str(values.get("GFS_WEB_ACCESS_TOKEN_FILE") or "").strip()
    if direct and secret_file:
        raise ValueError(
            "set only one of GFS_WEB_ACCESS_TOKEN or GFS_WEB_ACCESS_TOKEN_FILE",
        )
    if not secret_file:
        return direct
    path = Path(secret_file).resolve()
    if not path.is_file():
        raise ValueError("GFS Web access-token file does not exist")
    if path.stat().st_size > 4096:
        raise ValueError("GFS Web access-token file is too large")
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError("GFS Web access-token file is not readable UTF-8") from exc


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _normalize_host(value: str) -> str:
    raw = value.strip().rstrip(".")
    if not raw or any(char in raw for char in "/?#@"):
        raise ValueError("invalid allowed Web host")
    try:
        host = raw.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("invalid allowed Web host") from exc
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        labels = host.split(".")
        if len(host) > 253 or any(not HOST_LABEL.fullmatch(label) for label in labels):
            raise ValueError("invalid allowed Web host")
        return host


class WebAccessPolicy:
    """Process-local session auth; remote mode assumes a trusted TLS proxy."""

    def __init__(
        self, *, remote: bool = False, access_token: str = "",
        allowed_hosts: tuple[str, ...] = (),
        session_absolute_seconds: float = 8 * 60 * 60,
        session_idle_seconds: float = 30 * 60,
        max_sessions: int = 64,
        login_window_seconds: float = 60,
        client_failure_limit: int = 5,
        global_failure_limit: int = 100,
        max_rate_clients: int = 1024,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.remote = bool(remote)
        self._access_digest = b""
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._sessions: dict[bytes, tuple[float, float]] = {}
        self._client_failures: dict[str, deque[float]] = {}
        self._global_failures: deque[float] = deque()
        self._identity_salt = secrets.token_bytes(32)
        numeric = {
            "session_absolute_seconds": session_absolute_seconds,
            "session_idle_seconds": session_idle_seconds,
            "login_window_seconds": login_window_seconds,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0
            for value in numeric.values()
        ):
            raise ValueError("Web security durations must be finite and positive")
        bounded = {
            "max_sessions": max_sessions,
            "client_failure_limit": client_failure_limit,
            "global_failure_limit": global_failure_limit,
            "max_rate_clients": max_rate_clients,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in bounded.values()
        ):
            raise ValueError("Web security limits must be positive integers")
        if session_idle_seconds > session_absolute_seconds:
            raise ValueError("session idle timeout must not exceed absolute timeout")
        if client_failure_limit > global_failure_limit:
            raise ValueError("client failure limit must not exceed global failure limit")
        self.session_absolute_seconds = float(session_absolute_seconds)
        self.session_idle_seconds = float(session_idle_seconds)
        self.max_sessions = min(max_sessions, 1000)
        self.login_window_seconds = float(login_window_seconds)
        self.client_failure_limit = client_failure_limit
        self.global_failure_limit = global_failure_limit
        self.max_rate_clients = min(max_rate_clients, 10_000)
        if self.remote:
            token = str(access_token).strip()
            if (
                token.lower() in PLACEHOLDER_TOKENS
                or len(token) < 32
                or any(char.isspace() or not char.isprintable() for char in token)
                or len(set(token)) < 8
            ):
                raise ValueError(
                    "remote Web access requires GFS_WEB_ACCESS_TOKEN with at least "
                    "32 high-entropy characters",
                )
            normalized = tuple(dict.fromkeys(
                _normalize_host(host) for host in allowed_hosts if str(host).strip()
            ))
            if not normalized:
                raise ValueError("remote Web access requires an explicit allowed host")
            if any(host in {"0.0.0.0", "::"} for host in normalized):
                raise ValueError("wildcard bind addresses are not valid public Host values")
            self.allowed_hosts = frozenset(normalized)
            self._access_digest = hashlib.sha256(token.encode("utf-8")).digest()
        else:
            self.allowed_hosts = frozenset({"localhost", "127.0.0.1", "::1"})

    @staticmethod
    def request_host(environ: dict) -> str:
        authority = str(
            environ.get("HTTP_HOST") or environ.get("SERVER_NAME") or "",
        ).strip()
        try:
            hostname = urlsplit("//" + authority).hostname or ""
        except ValueError as exc:
            raise ValueError("invalid Host header") from exc
        return _normalize_host(hostname)

    def host_allowed(self, environ: dict) -> bool:
        try:
            host = self.request_host(environ)
        except ValueError:
            return False
        if self.remote:
            return host in self.allowed_hosts
        return is_loopback_host(host)

    def secure_transport(self, environ: dict) -> bool:
        if not self.remote:
            return True
        return str(environ.get("HTTP_X_FORWARDED_PROTO", "")).strip().lower() == "https"

    def authenticate_token(self, candidate: str) -> bool:
        if not self.remote:
            return True
        observed = hashlib.sha256(str(candidate).encode("utf-8")).digest()
        return secrets.compare_digest(observed, self._access_digest)

    def _client_key(self, environ: dict) -> str:
        values = []
        if self.remote:
            values.extend(
                item.strip()
                for item in str(environ.get("HTTP_X_FORWARDED_FOR", "")).split(",")
                if item.strip()
            )
        values.append(str(environ.get("REMOTE_ADDR", "")).strip())
        canonical = "unknown"
        for value in values:
            try:
                canonical = ipaddress.ip_address(value).compressed
                break
            except ValueError:
                continue
        return hashlib.blake2b(
            canonical.encode("ascii"), key=self._identity_salt, digest_size=16,
        ).hexdigest()

    def _prune_failures(self, now: float) -> None:
        cutoff = now - self.login_window_seconds
        while self._global_failures and self._global_failures[0] <= cutoff:
            self._global_failures.popleft()
        empty = []
        for key, failures in self._client_failures.items():
            while failures and failures[0] <= cutoff:
                failures.popleft()
            if not failures:
                empty.append(key)
        for key in empty:
            self._client_failures.pop(key, None)
        if len(self._client_failures) > self.max_rate_clients:
            oldest = sorted(
                self._client_failures,
                key=lambda key: self._client_failures[key][-1],
            )[:len(self._client_failures) - self.max_rate_clients]
            for key in oldest:
                self._client_failures.pop(key, None)

    def authenticate_login(self, environ: dict, candidate: str) -> tuple[str, int]:
        """Return accepted/invalid/rate_limited and a Retry-After value."""
        if not self.remote:
            return "accepted", 0
        now = float(self._clock())
        client_key = self._client_key(environ)
        with self._lock:
            self._prune_failures(now)
            client = self._client_failures.setdefault(client_key, deque())
            blocked = []
            if len(client) >= self.client_failure_limit:
                blocked.append(client[0])
            if len(self._global_failures) >= self.global_failure_limit:
                blocked.append(self._global_failures[0])
            if blocked:
                retry_after = max(
                    1, math.ceil(max(
                        timestamp + self.login_window_seconds - now
                        for timestamp in blocked
                    )),
                )
                return "rate_limited", retry_after
            if self.authenticate_token(candidate):
                self._client_failures.pop(client_key, None)
                return "accepted", 0
            client.append(now)
            self._global_failures.append(now)
            return "invalid", 0

    def _clean_sessions(self, now: float) -> None:
        expired = [
            digest for digest, (created, last_seen) in self._sessions.items()
            if (
                now - created >= self.session_absolute_seconds
                or now - last_seen >= self.session_idle_seconds
            )
        ]
        for digest in expired:
            self._sessions.pop(digest, None)

    @staticmethod
    def _session_digest(value: str) -> bytes:
        return hashlib.sha256(value.encode("utf-8")).digest()

    @staticmethod
    def _cookie_value(environ: dict) -> str:
        cookie = SimpleCookie()
        try:
            cookie.load(str(environ.get("HTTP_COOKIE", "")))
        except Exception:
            return ""
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else ""

    def authenticated(self, environ: dict) -> bool:
        if not self.remote:
            return True
        value = self._cookie_value(environ)
        if not value:
            return False
        digest = self._session_digest(value)
        now = float(self._clock())
        with self._lock:
            self._clean_sessions(now)
            session = self._sessions.get(digest)
            if session is None:
                return False
            self._sessions[digest] = (session[0], now)
            return True

    def session_cookie_header(self) -> str:
        if not self.remote:
            raise RuntimeError("local Web mode does not issue an auth cookie")
        value = secrets.token_urlsafe(48)
        digest = self._session_digest(value)
        now = float(self._clock())
        with self._lock:
            self._clean_sessions(now)
            while len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions, key=lambda key: self._sessions[key][0])
                self._sessions.pop(oldest, None)
            self._sessions[digest] = (now, now)
        return (
            f"{SESSION_COOKIE}={value}; Path=/; HttpOnly; Secure; "
            f"SameSite=Strict; Max-Age={int(self.session_absolute_seconds)}"
        )

    def revoke_session(self, environ: dict) -> bool:
        if not self.remote:
            return False
        value = self._cookie_value(environ)
        if not value:
            return False
        with self._lock:
            return self._sessions.pop(self._session_digest(value), None) is not None

    @staticmethod
    def clear_session_cookie_header() -> str:
        return (
            f"{SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Strict; "
            "Max-Age=0"
        )

    def public_summary(self) -> dict:
        return {
            "mode": "remote_authenticated" if self.remote else "local_loopback",
            "allowed_hosts": sorted(self.allowed_hosts),
            "tls_proxy_required": self.remote,
            "credentials_available": bool(self._access_digest) if self.remote else None,
            "session_absolute_seconds": (
                int(self.session_absolute_seconds) if self.remote else None
            ),
            "session_idle_seconds": (
                int(self.session_idle_seconds) if self.remote else None
            ),
            "max_sessions": self.max_sessions if self.remote else None,
            "login_rate_limit": ({
                "client_failures": self.client_failure_limit,
                "global_failures": self.global_failure_limit,
                "window_seconds": int(self.login_window_seconds),
            } if self.remote else None),
        }
