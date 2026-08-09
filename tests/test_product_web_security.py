import pytest

from src.product.web_security import (
    SESSION_COOKIE, WebAccessPolicy, is_loopback_host, resolve_web_access_token,
)


def _environ(
    host="studio.example", *, proto="https", cookie="", remote_addr="203.0.113.4",
    forwarded_for="",
):
    return {
        "HTTP_HOST": host,
        "HTTP_X_FORWARDED_PROTO": proto,
        "HTTP_COOKIE": cookie,
        "REMOTE_ADDR": remote_addr,
        "HTTP_X_FORWARDED_FOR": forwarded_for,
    }


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_loopback_detection(host):
    assert is_loopback_host(host)


def test_remote_policy_requires_high_entropy_token_and_explicit_host():
    with pytest.raises(ValueError, match="32"):
        WebAccessPolicy(remote=True, access_token="short", allowed_hosts=("studio.example",))
    with pytest.raises(ValueError, match="32"):
        WebAccessPolicy(
            remote=True,
            access_token="replace_with_32_or_more_random_characters",
            allowed_hosts=("studio.example",),
        )
    with pytest.raises(ValueError, match="high-entropy"):
        WebAccessPolicy(
            remote=True, access_token="x" * 64, allowed_hosts=("studio.example",),
        )
    valid_token = "test-only-GFS-access-9F3kLm2Q7Vr5"
    with pytest.raises(ValueError, match="allowed host"):
        WebAccessPolicy(remote=True, access_token=valid_token)
    with pytest.raises(ValueError, match="wildcard"):
        WebAccessPolicy(
            remote=True, access_token=valid_token, allowed_hosts=("0.0.0.0",),
        )


def test_remote_policy_enforces_host_https_token_and_secure_cookie():
    policy = WebAccessPolicy(
        remote=True, access_token="operator-secret-" + "x" * 32,
        allowed_hosts=("Studio.Example.",),
    )
    assert policy.host_allowed(_environ())
    assert not policy.host_allowed(_environ(host="attacker.example"))
    assert policy.secure_transport(_environ())
    assert not policy.secure_transport(_environ(proto="http"))
    assert policy.authenticate_token("operator-secret-" + "x" * 32)
    assert not policy.authenticate_token("wrong-" + "x" * 32)
    header = policy.session_cookie_header()
    assert all(value in header for value in (
        "__Host-", "HttpOnly", "Secure", "SameSite=Strict", "Max-Age=",
    ))
    value = header.split(";", 1)[0]
    assert policy.authenticated(_environ(cookie=value))
    assert not policy.authenticated(_environ(cookie=f"{SESSION_COOKIE}=wrong"))
    assert "operator-secret" not in repr(policy.public_summary())


def test_sessions_are_independent_bounded_expiring_and_individually_revocable():
    clock = FakeClock()
    policy = WebAccessPolicy(
        remote=True, access_token="session-test-token-" + "x" * 32,
        allowed_hosts=("studio.example",), clock=clock,
        session_absolute_seconds=20, session_idle_seconds=5, max_sessions=2,
    )
    first = policy.session_cookie_header().split(";", 1)[0]
    second = policy.session_cookie_header().split(";", 1)[0]
    assert first != second
    assert policy.authenticated(_environ(cookie=first))
    assert policy.authenticated(_environ(cookie=second))
    assert policy.revoke_session(_environ(cookie=first))
    assert not policy.authenticated(_environ(cookie=first))
    assert policy.authenticated(_environ(cookie=second))

    third = policy.session_cookie_header().split(";", 1)[0]
    fourth = policy.session_cookie_header().split(";", 1)[0]
    assert not policy.authenticated(_environ(cookie=second))
    assert policy.authenticated(_environ(cookie=third))
    clock.advance(5)
    assert not policy.authenticated(_environ(cookie=third))
    assert not policy.authenticated(_environ(cookie=fourth))
    assert "Max-Age=0" in policy.clear_session_cookie_header()


def test_session_absolute_expiry_is_not_extended_by_activity():
    clock = FakeClock()
    policy = WebAccessPolicy(
        remote=True, access_token="absolute-test-token-" + "x" * 32,
        allowed_hosts=("studio.example",), clock=clock,
        session_absolute_seconds=10, session_idle_seconds=6,
    )
    cookie = policy.session_cookie_header().split(";", 1)[0]
    clock.advance(5)
    assert policy.authenticated(_environ(cookie=cookie))
    clock.advance(5)
    assert not policy.authenticated(_environ(cookie=cookie))


def test_login_rate_limit_is_client_scoped_global_and_time_bounded():
    clock = FakeClock()
    token = "rate-limit-test-token-" + "x" * 32
    policy = WebAccessPolicy(
        remote=True, access_token=token, allowed_hosts=("studio.example",),
        clock=clock, login_window_seconds=10,
        client_failure_limit=2, global_failure_limit=4,
    )
    client_a = _environ(forwarded_for="198.51.100.1")
    assert policy.authenticate_login(client_a, "wrong")[0] == "invalid"
    assert policy.authenticate_login(client_a, "wrong")[0] == "invalid"
    decision, retry_after = policy.authenticate_login(client_a, token)
    assert decision == "rate_limited" and retry_after == 10

    client_b = _environ(forwarded_for="198.51.100.2")
    client_c = _environ(forwarded_for="198.51.100.3")
    assert policy.authenticate_login(client_b, "wrong")[0] == "invalid"
    assert policy.authenticate_login(client_c, "wrong")[0] == "invalid"
    other = _environ(forwarded_for="198.51.100.4")
    assert policy.authenticate_login(other, token)[0] == "rate_limited"
    clock.advance(10)
    assert policy.authenticate_login(client_a, token) == ("accepted", 0)
    summary = policy.public_summary()
    assert "198.51.100" not in repr(summary)
    assert "active_sessions" not in summary


def test_local_policy_never_requires_or_exposes_credentials():
    policy = WebAccessPolicy()
    assert policy.host_allowed(_environ(host="127.0.0.1", proto="http"))
    assert policy.secure_transport(_environ(host="127.0.0.1", proto="http"))
    assert policy.authenticated({})
    assert policy.public_summary()["credentials_available"] is None


def test_access_token_file_is_supported_without_persisting_value(tmp_path):
    secret = tmp_path / "gfs-token"
    secret.write_text("file-secret-" + "x" * 32 + "\n", encoding="utf-8")
    assert resolve_web_access_token({
        "GFS_WEB_ACCESS_TOKEN_FILE": str(secret),
    }) == "file-secret-" + "x" * 32
    with pytest.raises(ValueError, match="only one"):
        resolve_web_access_token({
            "GFS_WEB_ACCESS_TOKEN": "direct-" + "x" * 32,
            "GFS_WEB_ACCESS_TOKEN_FILE": str(secret),
        })
    with pytest.raises(ValueError, match="does not exist"):
        resolve_web_access_token({
            "GFS_WEB_ACCESS_TOKEN_FILE": str(tmp_path / "missing"),
        })
