from scripts.verify_web_accessibility import verify_web_accessibility


def test_accessibility_verifier_is_honest_and_zero_call():
    report = verify_web_accessibility()
    assert report["passed"]
    assert report["status"] == "passed_automated_contract"
    assert report["external_calls_made"] is False
    assert report["matches_executed"] == 0
    assert all(report["checks"].values())
    assert report["contrast_ratios"]["focus_indicator"] >= 3.0
    assert min(
        ratio
        for name, ratio in report["contrast_ratios"].items()
        if name != "focus_indicator"
    ) >= 4.5
    assert any("not an external WCAG" in item for item in report["limitations"])
    assert any("accessibility tree" in item for item in report["limitations"])
