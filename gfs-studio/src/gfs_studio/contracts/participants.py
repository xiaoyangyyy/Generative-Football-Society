from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ParticipantSessions(Protocol):
    def register(self, participant_id: str, role: str, moderator_id: str, capability_token: str, *, consent: bool) -> dict: ...

    def begin(self, participant_id: str, capability_token: str) -> dict: ...

    def session_view(self, participant_id: str, capability_token: str) -> dict: ...

    def status(self) -> dict: ...


@runtime_checkable
class ParticipantDelivery(Protocol):
    def current(self, participant_id: str, capability_token: str) -> dict: ...

    def submit(self, participant_id: str, capability_token: str, submission_id: str, answers: dict) -> dict: ...


@runtime_checkable
class ParticipantScoring(Protocol):
    def score_receipt(self, receipt: dict) -> dict: ...

    def score_participant(self, participant_id: str) -> dict: ...

    def evaluate(self) -> dict: ...
