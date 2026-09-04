import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_supply_chain_sbom import DEFAULT_OUTPUT, LOCK, expected_sbom
from src.infrastructure.supply_chain import parse_hashed_lock


ROOT = Path(__file__).resolve().parents[1]


def test_target_lock_is_exact_hashed_and_matches_direct_contract():
    packages = parse_hashed_lock(LOCK.read_text(encoding="utf-8"))
    assert len(packages) == 64
    assert len({row["name"] for row in packages}) == 64
    assert all(row["sha256"] for row in packages)
    observed = {row["name"]: row["version"].split("+", 1)[0] for row in packages}
    contract = json.loads(
        (ROOT / "data/evaluation/dependency_lock_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        observed.get(name) == version
        for name, version in contract["direct_dependencies"].items()
    )
    assert observed["tzdata"] == "2026.3"
    assert observed["colorama"] == "0.4.6"


def test_cyclonedx_sbom_is_deterministic_and_covers_runtime_closure():
    stored = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert stored == expected_sbom()
    assert stored["bomFormat"] == "CycloneDX"
    assert stored["specVersion"] == "1.6"
    assert len(stored["components"]) == 64
    assert len(stored["dependencies"][0]["dependsOn"]) == 64
    assert len({row["purl"] for row in stored["components"]}) == 64


def test_lock_parser_rejects_unhashed_and_duplicate_packages():
    with pytest.raises(ValueError, match="no SHA-256"):
        parse_hashed_lock("demo==1.0")
    duplicated = """demo==1.0 --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
demo==1.0 --hash=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"""
    with pytest.raises(ValueError, match="duplicate"):
        parse_hashed_lock(duplicated)


def test_sbom_cli_check_is_standalone_and_read_only():
    before = DEFAULT_OUTPUT.read_bytes()
    completed = subprocess.run(
        [sys.executable, "scripts/build_supply_chain_sbom.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["status"] == "current"
    assert DEFAULT_OUTPUT.read_bytes() == before
