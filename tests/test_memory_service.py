import pytest

from src.simulation.memory_service import attribute_delayed_utility, build_provenance, memory_stream_view


def test_memory_service_preserves_provenance_and_order():
    provenance = build_provenance("match", [1], [2], 3, None)
    assert provenance["causal_parent_ids"] == ["1"]
    records = [{"id": "b", "created_step": 2}, {"id": "a", "created_step": 1}]
    assert [row["content"] for row in memory_stream_view(records, [])] == ["", ""]
    assert attribute_delayed_utility(records, ["a"], 0.5) == 1
    assert records[1]["downstream_utility_sum"] == 0.5
    with pytest.raises(ValueError):
        attribute_delayed_utility(records, ["a"], float("nan"))
