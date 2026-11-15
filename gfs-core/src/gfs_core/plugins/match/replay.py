from gfs_core.kernel.identity import content_identity


def request_payload(home_profile, away_profile, config, signals, options) -> dict:
    return {
        "home_profile": home_profile,
        "away_profile": away_profile,
        "config": config,
        "signals": signals,
        "options": dict(options),
    }


def checkpoint_payload(request, next_tick, state, runtime, *, status) -> dict:
    request = _plain(request)
    return {
        "schema_version": 1,
        "request": request,
        "request_sha256": content_identity(request),
        "next_tick": next_tick,
        "state_sha256": content_identity({
            "state": state,
            "runtime": runtime,
        }),
        "status": status,
    }


def replay_target(document, request) -> int:
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported micro checkpoint")
    if document.get("request_sha256") != content_identity(_plain(request)):
        raise ValueError("micro checkpoint request identity mismatch")
    target = document.get("next_tick")
    if isinstance(target, bool) or not isinstance(target, int) or target < 0:
        raise ValueError("micro checkpoint tick is invalid")
    if len(document.get("state_sha256", "")) != 64:
        raise ValueError("micro checkpoint state identity is invalid")
    return target


def verify_checkpoint(document, next_tick, state, runtime) -> None:
    expected = document["state_sha256"]
    actual = content_identity({"state": state, "runtime": runtime})
    if actual != expected:
        raise ValueError(f"micro checkpoint replay diverged at tick {next_tick}")


def _plain(value):
    from gfs_core.kernel.identity import canonical_bytes
    import json

    return json.loads(canonical_bytes(value))
