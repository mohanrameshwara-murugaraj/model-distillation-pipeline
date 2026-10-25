import json

from distill_lab.report import make_three_axis_report


def test_combined_three_axis_report(tmp_path):
    source = tmp_path / "benchmark.json"
    econ = tmp_path / "economics.json"
    dest = tmp_path / "report.md"
    source.write_text(json.dumps({"sample_count": 2, "summaries": {
        "Teacher": {"accuracy": 1, "macro_f1": 1,
                    "p50_latency_ms": 50, "p95_latency_ms": 80},
        "Student": {"accuracy": .5, "macro_f1": .4,
                    "p50_latency_ms": 20, "p95_latency_ms": 30},
    }}))
    econ.write_text(json.dumps({
        "at_requested_volume": {"requests_per_day": 1000,
            "teacher_usd_per_1000": 5,
            "student_router_usd_per_1000": 3.0},
        "assumptions": {"student_variable_usd_per_request": 0},
        "fixed_daily_usd_including_idle": 2,
        "break_even_requests_per_day": 501,
        "break_even_within_measured_capacity": True,
        "data_provenance": {"escalation_basis": "measured"},
    }))
    report = make_three_axis_report(str(source), str(econ), str(dest))
    assert report["break_even_requests_per_day"] == 501
    assert "Gold accuracy" in dest.read_text()
    assert "Not measured" in dest.read_text()
