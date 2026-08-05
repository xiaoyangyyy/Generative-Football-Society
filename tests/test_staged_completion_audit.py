from scripts.audit_staged_completion import calibration_is_noop


def test_calibration_completion_uses_authoritative_action_field():
    assert calibration_is_noop({"action": "no_op_freeze_parameters"})
    assert not calibration_is_noop({"decision": "no_op_freeze_parameters"})
    assert not calibration_is_noop({"action": "run_bounded_search"})
