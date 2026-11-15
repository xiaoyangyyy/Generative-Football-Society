import json
import shutil
from pathlib import Path

import pytest

from gfs_core.kernel import PluginKernel
from gfs_studio.bootstrap.builtin_plugins import build_participants_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN = "test-capability-abcdefgh-12345678-unique"


def registry(root: Path):
    frozen = root / "evidence" / "frozen"
    frozen.mkdir(parents=True)
    for name in (
        "product_value_validation_protocol_v1.json",
        "product_value_case_packs_v1.json",
        "product_value_scoring_seal_v1.json",
    ):
        shutil.copy(PROJECT_ROOT / "evidence" / "frozen" / name, frozen / name)
    kernel = PluginKernel(build_participants_registry(root))
    kernel.start(["gfs.studio.participants"])
    return kernel, kernel.context.require("participants-bundle")


def answers(case_pack):
    intervention = "wide_overload" if case_pack == "case_pack_alpha" else "controlled_possession"
    return {
        "recommended_intervention": intervention,
        "supported_claim": "best_frozen_simulator_branch",
        "next_evidence_action": "preregistered_paired_validation",
    }


def test_participant_delivery_idempotency_and_blinded_scoring(tmp_path):
    kernel, participants = registry(tmp_path)
    try:
        sessions = participants["sessions"]
        assert sessions.status()["registered_participants"] == 0
        registered = sessions.register(
            "participant-testcase01", "football_analyst", "moderator-testcase01", TOKEN, consent=True,
        )
        assert registered["sequence"] in {"AB", "BA"}
        with pytest.raises(ValueError, match="training"):
            sessions.begin("participant-testcase01", TOKEN)
        sessions.complete_training("participant-testcase01", TOKEN)
        sessions.begin("participant-testcase01", TOKEN)
        first_packet = participants["delivery"].current("participant-testcase01", TOKEN)
        assert "scoring_keys" not in json.dumps(first_packet)
        first = participants["delivery"].submit(
            "participant-testcase01", TOKEN, "submission-one", answers(first_packet["case_pack"]),
        )
        replay = participants["delivery"].submit(
            "participant-testcase01", TOKEN, "submission-one", answers(first_packet["case_pack"]),
        )
        assert replay == first
        with pytest.raises(ValueError, match="different content"):
            participants["delivery"].submit(
                "participant-testcase01", TOKEN, "submission-one", {},
            )
        sessions.begin("participant-testcase01", TOKEN)
        second_packet = participants["delivery"].current("participant-testcase01", TOKEN)
        assert second_packet["case_pack"] != first_packet["case_pack"]
        participants["delivery"].submit(
            "participant-testcase01", TOKEN, "submission-two", answers(second_packet["case_pack"]),
        )
        sessions.attest("participant-testcase01", "moderator-testcase01")
        scored = participants["scoring"].score_participant("participant-testcase01")
        assert scored["valid_for_analysis"] is True
        assert scored["accuracy_delta"] == 0.0
        evaluation = participants["scoring"].evaluate()
        assert evaluation["valid_participants"] == 1
        assert evaluation["study_complete"] is False
        assert evaluation["user_value_validated"] is False
        persisted = (tmp_path / "data" / "persistence" / "participant_sessions.json").read_text()
        assert TOKEN not in persisted
    finally:
        kernel.stop()


def test_empty_study_does_not_claim_value():
    kernel = PluginKernel(build_participants_registry(PROJECT_ROOT))
    try:
        kernel.start(["gfs.studio.participants"])
        participants = kernel.context.require("participants-bundle")
        assert participants["sessions"].status()["registered_participants"] == 0
        evaluation = participants["scoring"].evaluate()
        assert evaluation["valid_participants"] == 0
        assert evaluation["study_complete"] is False
        assert evaluation["user_value_validated"] is False
    finally:
        kernel.stop()
