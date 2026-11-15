from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscriptions: dict[str, list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(
        self, topic: str, callback: Callable[[Any], None],
    ) -> Callable[[], None]:
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("event topic must be a non-empty string")
        if not callable(callback):
            raise TypeError("event callback must be callable")
        self._subscriptions[topic].append(callback)
        active = True

        def unsubscribe() -> None:
            nonlocal active
            if not active:
                return
            active = False
            callbacks = self._subscriptions.get(topic)
            if callbacks is None:
                return
            try:
                callbacks.remove(callback)
            except ValueError:
                pass
            if not callbacks:
                self._subscriptions.pop(topic, None)

        return unsubscribe

    def publish(self, topic: str, payload: Any = None) -> None:
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("event topic must be a non-empty string")
        for callback in tuple(self._subscriptions.get(topic, ())):
            callback(payload)

    def clear(self) -> None:
        self._subscriptions.clear()
