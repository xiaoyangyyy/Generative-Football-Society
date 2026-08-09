import pytest

from src.product.web_security import (
    SESSION_COOKIE, WebAccessPolicy, is_loopback_host, resolve_web_access_token,
)


def _environ(host="studio.example", *, proto="https", cookie=""):
    return {
        "HTTP_HOST": host,
        "HTTP_X_FORWARDED_PROTO": proto,
        "HTTP_COOKIE": cookie,
    }


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
    assert "HttpOnly" in header and "Secure" in header and "SameSite=Strict" in header
    value = header.split(";", 1)[0]
    assert policy.authenticated(_environ(cookie=value))
    assert not policy.authenticated(_environ(cookie=f"{SESSION_COOKIE}=wrong"))
    assert "operator-secret" not in repr(policy.public_summary())


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
