"""Combine real evaluation quality/latency and explicit cost assumptions in one table."""
import json
from pathlib import Path


def make_three_axis_report(benchmark_json: str, economics_json: str,
                           output_file: str) -> dict:
    bench = json.loads(Path(benchmark_json).read_text(encoding="utf-8"))
    econ = json.loads(Path(economics_json).read_text(encoding="utf-8"))
    summaries = bench["summaries"]
    cost = econ["at_requested_volume"]
    settings = econ["assumptions"]
    n = cost["requests_per_day"]
    teacher_usd = cost["teacher_usd_per_1000"]
    student_usd = 1000 * (
        econ["fixed_daily_usd_including_idle"] / n
        + settings["student_variable_usd_per_request"]
    )
    router_usd = cost["student_router_usd_per_1000"]
    table = [
        "# Three-axis benchmark — measured quality/latency + modeled unit economics", "",
        f"**Evaluation samples per system:** {bench['sample_count']}.",
        f"**Pricing scenario volume:** {n:,} requests/day.",
        f"**Escalation-rate basis:** {econ['data_provenance']['escalation_basis']}.", "",
        "| System | Gold accuracy | Macro-F1 | p50 | p95 | USD/1,000 requests |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, price in [("Teacher", teacher_usd), ("Student", student_usd),
                        ("Router", router_usd)]:
        row = summaries.get(name)
        if row:
            accuracy = f"{row['accuracy']:.2%}"
            f1 = f"{row['macro_f1']:.3f}"
            p50 = f"{row['p50_latency_ms']:.1f} ms" if row['p50_latency_ms'] is not None else "N/A"
            p95 = f"{row['p95_latency_ms']:.1f} ms" if row['p95_latency_ms'] is not None else "N/A"
        else:
            accuracy = f1 = p50 = p95 = "Not measured"
        table.append(f"| {name} | {accuracy} | {f1} | {p50} | {p95} | ${price:.4f} |")
    be = econ["break_even_requests_per_day"]
    table += ["",
              f"**Break-even requests/day:** {be:,}" if be is not None
              else "**Break-even requests/day:** none under these assumptions",
              f"**Capacity check:** {econ['break_even_within_measured_capacity']}", "",
              "**Interpretation:** Quality/latency rows come from actual prediction files; "
              "costs are workload scenarios using measured teacher tokens, user-supplied GPU "
              "pricing, and assumed or measured fallback fraction. They are not bills.",
              "Pricing excludes unreported retries, tier/caching differences, taxes and other operational expenses."
              ]
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(table) + "\n", encoding="utf-8")
    return {"three_axis_file": str(target), "break_even_requests_per_day": be,
            "rows_with_measured_quality": list(summaries)}
