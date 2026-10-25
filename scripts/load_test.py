"""Measure served-router throughput and p50/p95 across concurrency levels.

Ensure the routing endpoint and warmed GPU are running before invoking.
No provider calls occur without this script being executed explicitly.
"""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import httpx

from distill_lab.io import read_jsonl
from distill_lab.metrics import percentile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080/classify")
    p.add_argument("--source", default="data/processed/validation.jsonl")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--acknowledge-possible-paid-fallback", action="store_true",
                   help="Explicitly acknowledge that served routing can bill Claude API")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--p95-slo-ms", type=float, required=True)
    p.add_argument("--out", default="reports/load_test.json")
    args = p.parse_args()
    if not args.acknowledge_possible_paid_fallback:
        raise ValueError("Route load test may incur paid Claude calls; pass --acknowledge-possible-paid-fallback")
    if args.limit < 1:
        raise ValueError("limit must be >= 1")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats >= 1 and warmup >= 0 required")
    rows = read_jsonl(args.source)[:args.limit]
    if not rows:
        raise ValueError("No input examples")
    # Client shared across threads with persistent connections.
    with httpx.Client(timeout=90.0) as client:
        def one(text):
            start = time.perf_counter()
            response = client.post(args.url, json={"text": text})
            elapsed = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            return {"latency_ms": elapsed, "escalated": bool(response.json().get("escalated"))}
        for i in range(args.warmup):
            one(rows[i % len(rows)]["text"])
        report = []
        for concurrency in args.concurrency:
            if concurrency < 1:
                raise ValueError("concurrency must be >= 1")
            queries = [r["text"] for r in rows] * args.repeats
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(pool.map(one, queries))
            duration = time.perf_counter() - started
            latencies = [r["latency_ms"] for r in results]
            p95 = percentile(latencies, 95)
            report.append({
                "concurrency": concurrency,
                "requests": len(results), "duration_s": round(duration, 3),
                "throughput_rps": round(len(results) / duration, 3),
                "p50_latency_ms": round(percentile(latencies, 50), 2),
                "p95_latency_ms": round(p95, 2),
                "meets_p95_slo": p95 <= args.p95_slo_ms,
                "escalation_rate": sum(r["escalated"] for r in results) / len(results),
            })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"slo_ms": args.p95_slo_ms, "runs": report}, indent=2),
                      encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
