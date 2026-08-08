from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from src.simulation.runtime import environment_override, environment_snapshot


def test_environment_overrides_are_nested_and_restored(monkeypatch):
    monkeypatch.setenv("GFS_CONTEXT_TEST", "process")
    with environment_override({"GFS_CONTEXT_TEST": "outer"}):
        assert environment_snapshot()["GFS_CONTEXT_TEST"] == "outer"
        with environment_override({"GFS_CONTEXT_TEST": "inner"}):
            assert environment_snapshot()["GFS_CONTEXT_TEST"] == "inner"
        assert environment_snapshot()["GFS_CONTEXT_TEST"] == "outer"
    assert environment_snapshot()["GFS_CONTEXT_TEST"] == "process"


def test_environment_overrides_do_not_leak_between_threads(monkeypatch):
    monkeypatch.setenv("GFS_CONTEXT_TEST", "process")
    barrier = Barrier(2)

    def read(value):
        with environment_override({"GFS_CONTEXT_TEST": value}):
            barrier.wait(timeout=2)
            return environment_snapshot()["GFS_CONTEXT_TEST"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(read, ("stable", "research"))) == {"stable", "research"}
    assert environment_snapshot()["GFS_CONTEXT_TEST"] == "process"
