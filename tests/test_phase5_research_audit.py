import json

import pytest

from scripts.audit_phase5_research_layers import _last_json


def test_last_json_ignores_validator_preamble():
    assert _last_json("loading...\n{\"ok\": false}\n") == {"ok": False}


def test_last_json_requires_payload():
    with pytest.raises(ValueError):
        _last_json("loading only")
