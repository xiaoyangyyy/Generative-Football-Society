from pathlib import Path

import pytest

from gfs_core.kernel import PluginKernel
from gfs_studio.bootstrap.builtin_plugins import build_product_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_product_bundle_compares_real_matches_without_causal_claim():
    kernel = PluginKernel(build_product_registry(PROJECT_ROOT))
    try:
        kernel.start(["gfs.studio.product"])
        product = kernel.context.require("product-bundle")
        assert set(product) == {"workflows", "manager", "comparison", "recovery", "tasks"}
        baseline = product["workflows"].micro_match("Brazil", "Argentina", 42)
        treatment = product["workflows"].micro_match("Brazil", "Argentina", 43)
        comparison = product["comparison"].compare(baseline, treatment)
        assert comparison["goal_delta"] == [
            treatment["score"][0] - baseline["score"][0],
            treatment["score"][1] - baseline["score"][1],
        ]
        assert comparison["causal_claim"] is False
        aggregate = product["comparison"].aggregate([comparison])
        assert aggregate["comparisons"] == 1
        assert aggregate["mean_deltas"]["goal_delta_0"] == comparison["goal_delta"][0]
    finally:
        kernel.stop()


def test_comparison_rejects_different_fixtures():
    kernel = PluginKernel(build_product_registry(PROJECT_ROOT))
    try:
        kernel.start(["gfs.studio.product.comparison"])
        comparison = kernel.context.require("comparison")
        result = {
            "home": "Brazil", "away": "Argentina", "score": [0, 3],
            "xg": [0.1, 1.2], "passes": [77, 93], "shots": [1, 7],
            "pass_completion": [0.7, 0.8],
        }
        with pytest.raises(ValueError, match="same teams"):
            comparison.compare(result, {**result, "away": "France"})
    finally:
        kernel.stop()
