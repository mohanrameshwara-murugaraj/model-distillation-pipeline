"""CLI: reproducible data, paid teacher, SFT data, evaluation, and economics."""
import argparse
import json
import os
from pathlib import Path

from .benchmark import benchmark
from .build_sft import build, build_gold_baseline
from .data import prepare
from .economics import EconomicsInputs, teacher_api_cost
from .evaluation import run_evaluation
from .io import read_jsonl
from .report import make_three_axis_report
from .teacher_runs import run_teacher


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(prog="distill", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download BANKING77 and create leak-safe splits")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-fraction", type=float, default=0.1)

    p = sub.add_parser("label-teacher", help="Pay Claude to annotate a chosen split")
    p.add_argument("--input", default="data/processed/train.jsonl")
    p.add_argument("--output", default="data/teacher/train.jsonl")
    p.add_argument("--labels", default="data/processed/labels.json")
    p.add_argument("--model", default=os.getenv("TEACHER_MODEL", "claude-sonnet-4-6"))
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--execute-paid", action="store_true")
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--input-price", type=float, required=True, help="USD per 1M input tokens")
    p.add_argument("--output-price", type=float, required=True, help="USD per 1M output tokens")

    p = sub.add_parser("build-sft", help="Build SFT from schema-valid teacher labels")
    p.add_argument("--train", default="data/processed/train.jsonl")
    p.add_argument("--validation", default="data/processed/validation.jsonl")
    p.add_argument("--labels", default="data/processed/labels.json")
    p.add_argument("--teacher", default="data/teacher/train.jsonl")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--require-gold-agreement", action="store_true",
                   help="Optional gold-filter ablation; default keeps all valid teacher predictions")

    p = sub.add_parser("build-gold-baseline", help="Gold-trained control on matching real inputs")
    p.add_argument("--train", default="data/processed/train.jsonl")
    p.add_argument("--validation", default="data/processed/validation.jsonl")
    p.add_argument("--labels", default="data/processed/labels.json")
    p.add_argument("--match-teacher", default=None)
    p.add_argument("--out", default="data/gold-baseline")

    p = sub.add_parser("score", help="Score one provider file vs gold for a pilot")
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels", default="data/processed/labels.json")

    p = sub.add_parser("evaluate", help="Live inference; final-test and spend guarded")
    p.add_argument("--source", default="data/processed/validation.jsonl")
    p.add_argument("--labels", default="data/processed/labels.json")
    p.add_argument("--output", required=True)
    p.add_argument("--provider", required=True, choices=["teacher", "student", "router"])
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--confirm-final-test", action="store_true")
    p.add_argument("--execute-paid", action="store_true")
    p.add_argument("--budget-usd", type=float, default=0)
    p.add_argument("--input-price", type=float, default=0)
    p.add_argument("--output-price", type=float, default=0)
    p.add_argument("--threshold", type=float, default=None)

    p = sub.add_parser("benchmark", help="Benchmark on identical gold-labeled samples")
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--router", default=None)
    p.add_argument("--labels", default="data/processed/labels.json")
    p.add_argument("--out", default="reports/benchmark")
    p.add_argument("--expected", type=int, default=None,
                   help="Use 3080 for full official BANKING77 test")

    p = sub.add_parser("report", help="Join real quality/latency with priced economics")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--economics", required=True)
    p.add_argument("--out", default="reports/three_axis.md")

    p = sub.add_parser("economics", help="Break-even from actual teacher usage and GPU pricing")
    p.add_argument("--teacher-predictions", required=True)
    p.add_argument("--teacher-input-price", required=True, type=float)
    p.add_argument("--teacher-output-price", required=True, type=float)
    p.add_argument("--gpu-hourly", type=float, required=True)
    p.add_argument("--ops-per-day", type=float, default=0)
    p.add_argument("--amortized-setup-per-day", type=float, default=0)
    p.add_argument("--student-variable-per-request", type=float, default=0)
    p.add_argument("--router-predictions", help="Derive real escalation fraction from measured router run")
    p.add_argument("--assumed-escalation-rate", type=float, default=None,
                   help="Scenario only: value 0..1, not a measured result")
    p.add_argument("--requests-per-day", type=int, required=True)
    p.add_argument("--acceptable-rps", type=float, default=None,
                   help="Measured throughput meeting p95 SLO; optional")
    p.add_argument("--out", default="reports/economics.json")

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.out, args.seed, args.valid_fraction)
    elif args.command == "label-teacher":
        result = run_teacher(
            input_file=args.input, output_file=args.output, labels_file=args.labels,
            model=args.model, limit=args.limit, execute_paid=args.execute_paid,
            budget_usd=args.budget_usd, input_price=args.input_price,
            output_price=args.output_price,
        )
    elif args.command == "build-sft":
        result = build(args.train, args.validation, args.labels, args.teacher,
                       args.out, args.require_gold_agreement)
    elif args.command == "build-gold-baseline":
        result = build_gold_baseline(args.train, args.validation, args.labels,
                                    args.out, args.match_teacher)
    elif args.command == "score":
        from .metrics import score_predictions
        from .taxonomy import load_labels
        result = score_predictions(read_jsonl(args.predictions), load_labels(args.labels))
    elif args.command == "evaluate":
        model = args.model or (
            os.getenv("VLLM_MODEL", "banking77-lora") if args.provider == "student"
            else os.getenv("TEACHER_MODEL", "claude-sonnet-4-6")
        )
        result = run_evaluation(
            source_file=args.source, labels_file=args.labels, output_file=args.output,
            provider=args.provider, model=model, base_url=args.base_url,
            concurrency=args.concurrency, limit=args.limit,
            execute_paid=args.execute_paid, confirm_final_test=args.confirm_final_test,
            input_price=args.input_price, output_price=args.output_price,
            budget_usd=args.budget_usd, threshold=args.threshold,
        )
    elif args.command == "benchmark":
        result = benchmark(args.teacher, args.student, args.labels, args.out,
                           args.expected, args.router)
    elif args.command == "report":
        result = make_three_axis_report(args.benchmark, args.economics, args.out)
    elif args.command == "economics":
        source = read_jsonl(args.teacher_predictions)
        successful = [r for r in source if r.get("status") == "ok"
                      and r.get("input_tokens") is not None
                      and r.get("output_tokens") is not None]
        if not successful:
            raise ValueError("Need real teacher usage records before running economics")
        mean_teacher_cost = sum(
            teacher_api_cost(r["input_tokens"], r["output_tokens"],
                             args.teacher_input_price, args.teacher_output_price)
            for r in successful
        ) / len(successful)
        if args.router_predictions and args.assumed_escalation_rate is not None:
            raise ValueError("Use measured OR assumed escalation, not both")
        if args.router_predictions:
            routes = read_jsonl(args.router_predictions)
            if routes and any(r.get("status") != "ok" for r in routes):
                raise ValueError("Router failures detected; cannot infer an honest escalation rate")
            if not routes:
                raise ValueError("Router predictions empty")
            escalation_rate = sum(bool(r.get("escalated")) for r in routes) / len(routes)
            basis = "measured"
        elif args.assumed_escalation_rate is not None:
            escalation_rate = args.assumed_escalation_rate
            basis = "assumed scenario, NOT measured"
        else:
            raise ValueError("Supply --router-predictions or --assumed-escalation-rate")
        model = EconomicsInputs(
            teacher_usd_per_request=mean_teacher_cost,
            gpu_usd_per_hour=args.gpu_hourly,
            operations_usd_per_day=args.ops_per_day,
            amortized_setup_usd_per_day=args.amortized_setup_per_day,
            student_variable_usd_per_request=args.student_variable_per_request,
            escalation_rate=escalation_rate, acceptable_rps=args.acceptable_rps,
        )
        result = model.as_report(args.requests_per_day)
        result["data_provenance"] = {
            "teacher_usage_samples": len(successful),
            "teacher_prediction_file": args.teacher_predictions,
            "escalation_basis": basis,
            "warning": "No throughput or GPU cost verified automatically. All given prices are user-supplied."
        }
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        raise AssertionError("Unhandled CLI command")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
