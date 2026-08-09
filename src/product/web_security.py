"""Authentication and trusted-proxy boundary for the Studio Web surface."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import secrets
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


SESSION_COOKIE = "gfs_studio_session"
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
    ) -> None:
        self.remote = bool(remote)
        self._access_digest = b""
        self._session_id = ""
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
            self._session_id = secrets.token_urlsafe(48)
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

    def authenticated(self, environ: dict) -> bool:
        if not self.remote:
            return True
        cookie = SimpleCookie()
        try:
            cookie.load(str(environ.get("HTTP_COOKIE", "")))
        except Exception:
            return False
        morsel = cookie.get(SESSION_COOKIE)
        return bool(
            morsel and secrets.compare_digest(morsel.value, self._session_id)
        )

    def session_cookie_header(self) -> str:
        if not self.remote:
            raise RuntimeError("local Web mode does not issue an auth cookie")
        return (
            f"{SESSION_COOKIE}={self._session_id}; Path=/; HttpOnly; Secure; "
            "SameSite=Strict"
        )

    def public_summary(self) -> dict:
        return {
            "mode": "remote_authenticated" if self.remote else "local_loopback",
            "allowed_hosts": sorted(self.allowed_hosts),
            "tls_proxy_required": self.remote,
            "credentials_available": bool(self._access_digest) if self.remote else None,
        }
