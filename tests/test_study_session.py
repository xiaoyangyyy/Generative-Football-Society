import hashlib
import io
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from wsgiref.util import setup_testing_defaults

import pytest

from scripts.product_value_study import analyze
from src.cli import build_parser
from src.infrastructure import LeaseUnavailable
from src.product.study_delivery import (
    CASE_PACK_RELATIVE,
    PROTOCOL_RELATIVE,
    SCORING_SEAL_RELATIVE,
    ProductValueStudyDelivery,
)
from src.product.study_session import (
    ParticipantSessionRuntime,
    ParticipantStudyWebApp,
    StudySessionError,
    create_participant_study_server,
    import_completed_session,
    provision_participant_session,
)


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "participant-capability-token-1234567890abcdef"
CORRECT = {
    "case_pack_alpha": {
        "recommended_intervention": "wide_overload",
        "supported_claim": "best_frozen_simulator_branch",
        "next_evidence_action": "preregistered_paired_validation",
    },
    "case_pack_beta": {
        "recommended_intervention": "controlled_possession",
        "supported_claim": "best_frozen_simulator_branch",
        "next_evidence_action": "preregistered_paired_validation",
    },
}


class Clock:
    def __init__(self):
        self.value = datetime(2026, 9, 5, 8, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


def _copy_authority(project):
    for relative in (
        PROTOCOL_RELATIVE, CASE_PACK_RELATIVE, SCORING_SEAL_RELATIVE,
    ):
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)


def _setup(tmp_path, *, participant_index=1):
    project = tmp_path / "project"
    sessions = tmp_path / "participant-sessions"
    evidence = tmp_path / "evidence-archive"
    _copy_authority(project)
    service = ProductValueStudyDelivery(project)
    registration = service.register(
        participant_id=f"participant-{participant_index:08d}",
        target_role="football_analyst",
        moderator_id="moderator-12345678",
        consent_recorded=True,
    )["registration"]
    clock = Clock()
    provisioned = provision_participant_session(
        project,
        registration_id=registration["registration_id"],
        session_root=sessions.resolve(),
        origin="http://127.0.0.1:8876",
        clock=clock,
        token_factory=lambda: TOKEN,
    )
    token = parse_qs(urlsplit(provisioned["launch_url"]).fragment)["access"][0]
    runtime = ParticipantSessionRuntime(
        project,
        Path(provisioned["session_file"]),
        evidence.resolve(),
        clock=clock,
    )
    return project, sessions, evidence, registration, provisioned, runtime, clock, token


def _request(
    app, method="GET", path="/", payload=None, *, token=None,
    host="127.0.0.1", forwarded_proto=None, remote_addr=None,
):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    environ = {}
    setup_testing_defaults(environ)
    environ.update({
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
        "HTTP_HOST": host,
    })
    if payload is not None:
        environ["CONTENT_TYPE"] = "application/json"
    if token is not None:
        environ["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    if forwarded_proto is not None:
        environ["HTTP_X_FORWARDED_PROTO"] = forwarded_proto
    if remote_addr is not None:
        environ["REMOTE_ADDR"] = remote_addr
    observed = {}

    def start_response(status, headers):
        observed["status"] = status
        observed["headers"] = dict(headers)

    observed["body"] = b"".join(app(environ, start_response))
    if observed["headers"]["Content-Type"].startswith("application/json"):
        observed["json"] = json.loads(observed["body"])
    return observed


def _complete_training(runtime, token):
    return runtime.complete_training(token, "simulator_local_comparison_only")


def _submit_current(runtime, clock, token, index):
    started = runtime.start_current(token)
    current = started["current"]
    case_pack = json.loads(runtime.session_file.read_text(encoding="utf-8"))[
        "conditions"
    ][current["position"] - 1]["case_pack"]
    clock.advance(10 + index)
    return runtime.submit_current(
        token,
        submission_id=f"SUB-{index:032x}",
        answers=CORRECT[case_pack],
    )


def test_session_reveals_only_current_condition_and_imports_real_receipts(tmp_path):
    (
        project, _, evidence, registration, provisioned, runtime, clock, token,
    ) = _setup(tmp_path)
    assert token == TOKEN
    assert TOKEN not in Path(provisioned["session_file"]).read_text(encoding="utf-8")
    with pytest.raises(StudySessionError, match="Valid session access"):
        runtime.view("wrong-capability")

    training = runtime.view(token)
    serialized = json.dumps(training)
    assert training["phase"] == "training"
    assert "participant_packet" not in serialized
    assert "case_pack_alpha" not in serialized
    assert "case_pack_beta" not in serialized
    ready = _complete_training(runtime, token)
    assert ready["phase"] == "ready"
    assert ready["current"] == {
        "position": 1, "state": "available", "maximum_seconds": 900,
    }

    first = runtime.start_current(token)
    first_state = json.loads(runtime.session_file.read_text(encoding="utf-8"))
    first_case = first_state["conditions"][0]["case_pack"]
    second_case = first_state["conditions"][1]["case_pack"]
    first_json = json.dumps(first)
    first_title = first_state["conditions"][0]["participant_packet"]["title"]
    second_title = first_state["conditions"][1]["participant_packet"]["title"]
    assert first_title in first_json
    assert second_title not in first_json
    assert first_case not in first_json
    assert second_case not in first_json
    assert registration["moderator_id"] not in first_json
    assert registration["participant_id"] not in first_json
    assert '"scoring_key"' not in first_json
    condition = first_state["conditions"][0]["condition"]
    assert first["current"]["presentation"]["computed_assistance"] is (
        condition == "gfs_studio"
    )

    clock.advance(37)
    first_result = runtime.submit_current(
        token,
        submission_id="SUB-00000000000000000000000000000001",
        answers=CORRECT[first_case],
    )
    assert first_result["status"] == "submitted"
    assert (evidence / first_result["receipt_sha256"]).is_file()
    duplicate = runtime.submit_current(
        token,
        submission_id="SUB-00000000000000000000000000000001",
        answers=CORRECT[first_case],
    )
    assert duplicate["status"] == "already_submitted"
    assert duplicate["receipt_sha256"] == first_result["receipt_sha256"]
    conflicting = dict(CORRECT[first_case])
    conflicting["supported_claim"] = "proven_real_world_tactic"
    with pytest.raises(StudySessionError, match="already used"):
        runtime.submit_current(
            token,
            submission_id="SUB-00000000000000000000000000000001",
            answers=conflicting,
        )
    assert "participant_packet" not in json.dumps(first_result["session"])

    second = runtime.start_current(token)
    assert second_title in json.dumps(second)
    assert first_title not in json.dumps(second)
    assert second_case not in json.dumps(second)
    clock.advance(53)
    second_result = runtime.submit_current(
        token,
        submission_id="SUB-00000000000000000000000000000002",
        answers=CORRECT[second_case],
    )
    assert second_result["session"]["phase"] == "completed"
    assert second_result["correctness_disclosed"] is False
    assert len(list(evidence.iterdir())) == 2

    with pytest.raises(ValueError, match="attestation"):
        import_completed_session(
            project,
            session_file=provisioned["session_file"],
            evidence_root=evidence.resolve(),
            observer_attested=False,
        )
    imported = import_completed_session(
        project,
        session_file=provisioned["session_file"],
        evidence_root=evidence.resolve(),
        observer_attested=True,
    )
    assert imported["status"] == "participant_record_imported"
    assert imported["scoring_performed"] is False
    assert import_completed_session(
        project,
        session_file=provisioned["session_file"],
        evidence_root=evidence.resolve(),
        observer_attested=True,
    )["status"] == "already_imported"
    result = analyze(
        project / "data/evaluation/product_value_validation_v1/participant_records.jsonl",
        project / "data/evaluation/product_value_validation_v1/session_registry.json",
        evidence,
        project / PROTOCOL_RELATIVE,
        project / CASE_PACK_RELATIVE,
        project / SCORING_SEAL_RELATIVE,
    )
    assert result["evidence_archive"]["verified_artifacts"] == 2
    assert result["registry_checks"]["all_records_registered"] is True


def test_timeout_advances_without_receipt_and_second_condition_can_finish(tmp_path):
    project, _, evidence, _, provisioned, runtime, clock, token = _setup(tmp_path)
    _complete_training(runtime, token)
    runtime.start_current(token)
    clock.advance(901)
    advanced = runtime.view(token)
    assert advanced["phase"] == "ready"
    assert advanced["current"]["position"] == 2
    state = json.loads(runtime.session_file.read_text(encoding="utf-8"))
    assert state["conditions"][0]["state"] == "timed_out"
    assert state["conditions"][0]["receipt_sha256"] is None
    second_case = state["conditions"][1]["case_pack"]
    runtime.start_current(token)
    clock.advance(25)
    runtime.submit_current(
        token,
        submission_id="SUB-00000000000000000000000000000003",
        answers=CORRECT[second_case],
    )
    import_completed_session(
        project,
        session_file=provisioned["session_file"],
        evidence_root=evidence.resolve(),
        observer_attested=True,
    )
    record = json.loads(
        (project / "data/evaluation/product_value_validation_v1/participant_records.jsonl")
        .read_text(encoding="utf-8")
    )
    assert record["conditions"][0]["completed"] is False
    assert record["conditions"][0]["duration_seconds"] == 900.0
    assert record["conditions"][1]["completed"] is True


def test_participant_web_has_narrow_capability_bound_surface(tmp_path):
    _, _, _, registration, _, runtime, _, token = _setup(tmp_path)
    app = ParticipantStudyWebApp(runtime)
    root = _request(app)
    document = root["body"].decode("utf-8")
    assert root["status"].startswith("200")
    assert root["headers"]["Cache-Control"] == "no-store"
    assert "Content-Security-Policy" in root["headers"]
    csp = root["headers"]["Content-Security-Policy"]
    nonce = re.search(r"script-src 'nonce-([^']+)'", csp).group(1)
    assert "unsafe-inline" not in csp
    assert f'nonce="{nonce}"' in document
    assert registration["participant_id"] not in document
    assert "scoring_key" not in document
    unauthorized = _request(app, path="/api/v1/session")
    assert unauthorized["status"].startswith("401")
    authorized = _request(app, path="/api/v1/session", token=token)
    assert authorized["status"].startswith("200")
    assert authorized["json"]["future_condition_exposed"] is False
    studio = _request(app, path="/api/v1/studio", token=token)
    assert studio["status"].startswith("404")
    hostile = _request(
        app, path="/api/v1/session", token=token, host="attacker.example",
    )
    assert hostile["status"].startswith("400")


def test_remote_surface_requires_loopback_proxy_https_and_exact_host(tmp_path):
    project, _, evidence, _, provisioned, runtime, _, token = _setup(tmp_path)
    app = ParticipantStudyWebApp(
        runtime, allow_remote=True, allowed_hosts=("study.example",),
    )
    direct = _request(
        app,
        path="/api/v1/session",
        token=token,
        host="study.example",
        forwarded_proto="https",
        remote_addr="198.51.100.20",
    )
    assert direct["status"].startswith("400")
    assert direct["json"]["error"]["code"] == "trusted_proxy_required"
    plaintext = _request(
        app,
        path="/api/v1/session",
        token=token,
        host="study.example",
        forwarded_proto="http",
        remote_addr="127.0.0.1",
    )
    assert plaintext["status"].startswith("426")
    accepted = _request(
        app,
        path="/api/v1/session",
        token=token,
        host="study.example",
        forwarded_proto="https",
        remote_addr="127.0.0.1",
    )
    assert accepted["status"].startswith("200")
    with pytest.raises(ValueError, match="bind to loopback"):
        create_participant_study_server(
            project,
            session_file=provisioned["session_file"],
            evidence_root=evidence.resolve(),
            host="0.0.0.0",
            allow_remote=True,
            allowed_hosts=("study.example",),
        )


def test_concurrent_submissions_commit_exactly_one_condition(tmp_path):
    _, _, evidence, _, _, runtime, clock, token = _setup(tmp_path)
    _complete_training(runtime, token)
    started = runtime.start_current(token)
    state = json.loads(runtime.session_file.read_text(encoding="utf-8"))
    answers = CORRECT[state["conditions"][0]["case_pack"]]
    clock.advance(10)

    def submit(index):
        try:
            return runtime.submit_current(
                token,
                submission_id=f"SUB-{index:032x}",
                answers=answers,
            )["status"]
        except StudySessionError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, (10, 11)))
    assert sorted(results) == ["condition_not_started", "submitted"]
    assert len(list(evidence.iterdir())) == 1
    assert started["current"]["position"] == 1


def test_session_paths_are_external_and_cli_commands_are_wired(tmp_path):
    project, _, _, registration, _, _, _, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        provision_participant_session(
            project,
            registration_id=registration["registration_id"],
            session_root=(project / "build/sessions").resolve(),
            origin="http://127.0.0.1:8876",
        )
    with pytest.raises(ValueError, match="HTTPS"):
        provision_participant_session(
            project,
            registration_id=registration["registration_id"],
            session_root=(tmp_path / "other-sessions").resolve(),
            origin="http://study.example",
        )
    with pytest.raises(ValueError, match="high-entropy"):
        provision_participant_session(
            project,
            registration_id=registration["registration_id"],
            session_root=(tmp_path / "weak-token-sessions").resolve(),
            origin="http://127.0.0.1:8876",
            token_factory=lambda: "a" * 64,
        )
    parser = build_parser()
    provision = parser.parse_args([
        "--base-dir", str(project), "studio", "value-study", "provision",
        "--registration-id", registration["registration_id"],
        "--session-root", str((tmp_path / "cli-sessions").resolve()),
    ])
    assert provision.func.__name__ == "cmd_studio_value_study_provision"
    serve = parser.parse_args([
        "--base-dir", str(project), "studio", "value-study", "serve",
        "--session-file", str((tmp_path / "session.json").resolve()),
        "--evidence-root", str((tmp_path / "evidence").resolve()),
    ])
    assert serve.func.__name__ == "cmd_studio_value_study_serve"
    imported = parser.parse_args([
        "--base-dir", str(project), "studio", "value-study", "import",
        "--session-file", str((tmp_path / "session.json").resolve()),
        "--evidence-root", str((tmp_path / "evidence").resolve()),
        "--attest-observed-session",
    ])
    assert imported.attest_observed_session is True


def test_participant_server_lease_prevents_duplicate_runtime(tmp_path):
    project, _, evidence, _, provisioned, _, _, _ = _setup(tmp_path)
    arguments = {
        "session_file": provisioned["session_file"],
        "evidence_root": evidence.resolve(),
        "host": "127.0.0.1",
        "port": 0,
    }
    server = create_participant_study_server(project, **arguments)
    try:
        assert server.server_port > 0
        with pytest.raises(LeaseUnavailable, match="already owned"):
            create_participant_study_server(project, **arguments)
    finally:
        server.server_close()
    replacement = create_participant_study_server(project, **arguments)
    replacement.server_close()


def test_receipt_name_is_exact_content_hash(tmp_path):
    _, _, evidence, _, _, runtime, clock, token = _setup(tmp_path)
    _complete_training(runtime, token)
    result = _submit_current(runtime, clock, token, 99)
    receipt = evidence / result["receipt_sha256"]
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == receipt.name


def test_completion_record_recovers_after_state_commit_crash(tmp_path):
    project, _, evidence, _, provisioned, runtime, clock, token = _setup(tmp_path)
    _complete_training(runtime, token)
    _submit_current(runtime, clock, token, 20)
    runtime.start_current(token)
    state = json.loads(runtime.session_file.read_text(encoding="utf-8"))
    second_case = state["conditions"][1]["case_pack"]
    clock.advance(20)

    def crash_after_state_commit(_state):
        raise RuntimeError("simulated record materialization crash")

    runtime._ensure_completion_record = crash_after_state_commit
    with pytest.raises(RuntimeError, match="materialization crash"):
        runtime.submit_current(
            token,
            submission_id="SUB-00000000000000000000000000000021",
            answers=CORRECT[second_case],
        )
    committed = json.loads(runtime.session_file.read_text(encoding="utf-8"))
    record_path = (
        runtime.session_file.parent / "records" / f"{committed['session_id']}.json"
    )
    assert committed["phase"] == "completed"
    assert not record_path.exists()

    recovered = ParticipantSessionRuntime(
        project, provisioned["session_file"], evidence.resolve(), clock=clock,
    )
    assert record_path.is_file()
    assert hashlib.sha256(record_path.read_bytes()).hexdigest() == (
        committed["completion_record_sha256"]
    )
    assert recovered.view(token)["phase"] == "completed"


def test_import_rejects_tampered_receipt_before_repository_write(tmp_path):
    project, _, evidence, _, provisioned, runtime, clock, token = _setup(tmp_path)
    _complete_training(runtime, token)
    first = _submit_current(runtime, clock, token, 30)
    _submit_current(runtime, clock, token, 31)
    (evidence / first["receipt_sha256"]).write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        import_completed_session(
            project,
            session_file=provisioned["session_file"],
            evidence_root=evidence.resolve(),
            observer_attested=True,
        )
    assert not (
        project / "data/evaluation/product_value_validation_v1/participant_records.jsonl"
    ).exists()


def test_import_rejects_case_authority_and_duration_drift(tmp_path):
    project, _, evidence, _, provisioned, runtime, clock, token = _setup(tmp_path)
    _complete_training(runtime, token)
    _submit_current(runtime, clock, token, 40)
    _submit_current(runtime, clock, token, 41)
    state_path = Path(provisioned["session_file"])
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["conditions"][0]["participant_packet"]["title"] = "tampered title"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="case authority drifted"):
        import_completed_session(
            project,
            session_file=state_path,
            evidence_root=evidence.resolve(),
            observer_attested=True,
        )

    state["conditions"][0]["participant_packet"]["title"] = (
        ProductValueStudyDelivery(project)
        .packet(state["registration"]["registration_id"])["conditions"][0]
        ["participant_packet"]["title"]
    )
    state["conditions"][0]["duration_seconds"] += 1.0
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="timing is invalid"):
        import_completed_session(
            project,
            session_file=state_path,
            evidence_root=evidence.resolve(),
            observer_attested=True,
        )
    assert not (
        project / "data/evaluation/product_value_validation_v1/participant_records.jsonl"
    ).exists()
