import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT.parent / "scripts" / "benchmark_runtime.py"


def test_cohort_benchmark_reports_real_work_and_metric_scope(capsys):
    spec = importlib.util.spec_from_file_location("gfs_benchmark_runtime", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--suite", "cohort", "--stop-at-tick", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["suite"] == "cohort"
    assert report["results"]["cohort"]["rows"] == 2
    assert report["results"]["cohort"]["matches"] == 2
    assert report["scope"] == "local_performance_not_production_sla"
    assert report["memory_metric"] == "tracemalloc_python_peak_not_process_rss"
