"""Canonical memory bookkeeping used by the SocietyAgent compatibility facade."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def build_provenance(
    source: str,
    causal_parent_ids: Iterable[str] | None,
    contradicts: Iterable[str] | None,
    valid_from,
    valid_until,
) -> dict:
    return {
        "source": str(source),
        "causal_parent_ids": [str(value) for value in (causal_parent_ids or [])],
        "contradicts": [str(value) for value in (contradicts or [])],
        "valid_from": valid_from,
        "valid_until": valid_until,
    }


def memory_stream_view(episodic: list[dict], procedural: list[dict]) -> list[dict]:
    merged = sorted(episodic + procedural, key=lambda record: record.get("created_step", 0))
    return [
        {
            "content": record.get("content", ""),
            "importance": record.get("importance", 0.0),
            "speaker": record.get("speaker"),
            "date": record.get("date"),
            "layer": record.get("layer", "episodic"),
            "salience": record.get("salience", 0.0),
        }
        for record in merged
    ]


def attribute_delayed_utility(records: Iterable[dict], memory_ids: Iterable[str], utility: float) -> int:
    value = float(utility)
    if not np.isfinite(value):
        raise ValueError("Memory utility must be finite")
    ids = {str(item) for item in memory_ids}
    updated = 0
    for record in records:
        if str(record.get("id")) not in ids:
            continue
        record["downstream_utility_sum"] = float(record.get("downstream_utility_sum", 0.0)) + value
        record["downstream_utility_count"] = int(record.get("downstream_utility_count", 0)) + 1
        updated += 1
    return updated
