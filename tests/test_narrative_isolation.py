"""Phase 5 — narrative isolation helpers (no full match sim)."""

from __future__ import annotations

import os

from src.match_engine.calibration.ablation import PIPELINE_PRESETS
from src.match_engine.calibration.narrative_isolation import (
    apply_benchmark_isolation,
    assert_isolated_env,
    expected_layer_flags,
    narrative_layers_enabled,
    restore_env,
)


def test_expected_layer_flags_m0():
    flags = expected_layer_flags("M0")
    assert flags["MATCH_MICRO"] == "1"
    assert flags["MATCH_COGNITIVE"] == "0"
    assert flags["MATCH_WORLD_MODEL"] == "0"
    assert flags["XG_SUPPLEMENT"] == "0"


def test_expected_layer_flags_m1_enables_wm():
    flags = expected_layer_flags("M1")
    assert flags["MATCH_WORLD_MODEL"] == "1"
    assert flags["MATCH_COGNITIVE"] == "0"


def test_expected_layer_flags_c1_enables_cognitive():
    flags = expected_layer_flags("C1")
    assert flags["MATCH_COGNITIVE"] == "1"
    assert flags["MATCH_WORLD_MODEL"] == "0"


def test_assert_isolated_env_blocks_leak():
    try:
        assert_isolated_env("M0", {"MATCH_COGNITIVE": "1", "MATCH_WORLD_MODEL": "0"})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "MATCH_COGNITIVE" in str(exc)


def test_assert_isolated_env_allows_m1():
    assert_isolated_env("M1", expected_layer_flags("M1"))


def test_narrative_layers_disabled_in_calibration_mode():
    old = os.environ.get("CALIBRATION_MODE")
    os.environ["CALIBRATION_MODE"] = "1"
    try:
        assert narrative_layers_enabled() is False
    finally:
        if old is None:
            os.environ.pop("CALIBRATION_MODE", None)
        else:
            os.environ["CALIBRATION_MODE"] = old


def test_apply_benchmark_isolation_roundtrip():
    saved = apply_benchmark_isolation("M0")
    try:
        assert os.environ.get("MATCH_MICRO") == "1"
        assert os.environ.get("MATCH_COGNITIVE") == "0"
    finally:
        restore_env(saved)


def test_all_pipeline_presets_have_expected_flags():
    for name in PIPELINE_PRESETS:
        flags = expected_layer_flags(name)
        assert flags["CALIBRATION_MODE"] == "1"
        assert flags["MATCH_MICRO"] == "1"
