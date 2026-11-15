import pytest

from gfs_core.kernel import (
    CapabilityError,
    PluginActivationError,
    PluginError,
    PluginKernel,
    PluginRegistry,
    PluginStateError,
)
from tests.helpers import StubPlugin


def test_kernel_starts_dependencies_and_stops_in_reverse_order():
    events = []
    alpha = StubPlugin(
        "example.alpha", provides=("alpha",), events=events,
    )

    class Beta(StubPlugin):
        def activate(self, context):
            assert context.require("alpha") is alpha
            super().activate(context)

    beta = Beta(
        "example.beta",
        requires=("example.alpha",),
        provides=("beta",),
        events=events,
    )
    kernel = PluginKernel(PluginRegistry([beta, alpha])).start(["example.beta"])
    assert kernel.running is True
    assert kernel.active_ids == ("example.alpha", "example.beta")
    assert kernel.context.capabilities == ("alpha", "beta")
    assert kernel.context.require("beta") is beta
    with pytest.raises(PluginStateError):
        kernel.start()
    kernel.stop()
    assert events == [
        "activate:example.alpha",
        "activate:example.beta",
        "deactivate:example.beta",
        "deactivate:example.alpha",
    ]
    assert kernel.running is False
    assert kernel.active_ids == ()
    assert kernel.context.capabilities == ()


def test_activation_failure_rolls_back_and_clears_context():
    events = []
    alpha = StubPlugin("example.alpha", provides=("alpha",), events=events)
    beta = StubPlugin(
        "example.beta",
        requires=("example.alpha",),
        events=events,
        fail_activate=True,
    )
    original_activate = alpha.activate

    def activate_with_subscription(context):
        original_activate(context)
        context.subscribe("event", lambda payload: None)

    alpha.activate = activate_with_subscription
    kernel = PluginKernel(PluginRegistry([alpha, beta]))
    with pytest.raises(PluginActivationError) as caught:
        kernel.start()
    assert caught.value.plugin_id == "example.beta"
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert kernel.running is False
    assert kernel.active_ids == ()
    assert kernel.context.capabilities == ()
    kernel.context.publish("event", {"value": 1})
    assert events == [
        "activate:example.alpha",
        "deactivate:example.alpha",
    ]


def test_stop_attempts_every_deactivation_and_can_restart():
    events = []
    alpha = StubPlugin(
        "example.alpha", provides=("alpha",), events=events,
        fail_deactivate=True,
    )
    beta = StubPlugin(
        "example.beta",
        requires=("example.alpha",),
        provides=("beta",),
        events=events,
    )
    kernel = PluginKernel(PluginRegistry([alpha, beta])).start()
    with pytest.raises(PluginError, match="failed to deactivate"):
        kernel.stop()
    assert events[-2:] == [
        "deactivate:example.beta",
        "deactivate:example.alpha",
    ]
    assert kernel.running is False
    alpha.fail_deactivate = False
    kernel.start()
    kernel.stop()
    assert kernel.context.capabilities == ()


def test_context_rejects_duplicate_or_missing_capabilities():
    alpha = StubPlugin("example.alpha", provides=("alpha",))
    kernel = PluginKernel(PluginRegistry([alpha])).start()
    with pytest.raises(CapabilityError, match="already provided"):
        kernel.context.provide("alpha", object())
    with pytest.raises(CapabilityError, match="not available"):
        kernel.context.require("missing")
    assert kernel.context.find("missing", "fallback") == "fallback"


def test_registry_rejects_unsupported_contract_version():
    plugin = StubPlugin("example.alpha", contract_version=2)
    with pytest.raises(ValueError, match="unsupported contract version"):
        PluginRegistry([plugin])
