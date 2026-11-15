from gfs_core.kernel import EventBus


def test_events_are_synchronous_ordered_and_unsubscribable():
    bus = EventBus()
    received = []
    unsubscribe_first = bus.subscribe("match", lambda payload: received.append(("first", payload)))
    bus.subscribe("match", lambda payload: received.append(("second", payload)))
    bus.publish("match", 42)
    assert received == [("first", 42), ("second", 42)]
    unsubscribe_first()
    unsubscribe_first()
    bus.publish("match", 43)
    assert received[-1] == ("second", 43)
    bus.clear()
    bus.publish("match", 44)
    assert len(received) == 3
