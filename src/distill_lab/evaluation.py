"""Concurrently benchmark real providers; explicit pay and final-test guards."""
from pathlib import Path
import concurrent.futures

from .economics import teacher_api_cost
from .io import read_jsonl, write_jsonl
from .metrics import score_predictions
from .providers import ClaudeTeacher, VLLMStudent
from .router import InferenceRouter
from .taxonomy import load_labels


def run_evaluation(
    *, source_file: str, labels_file: str, output_file: str,
    provider: str, model: str, base_url: str,
    concurrency: int = 1, limit: int | None = None,
    execute_paid: bool = False, confirm_final_test: bool = False,
    input_price: float = 0.0, output_price: float = 0.0,
    budget_usd: float = 0.0,
    threshold: float | None = None,
) -> dict:
    if concurrency < 1:
        raise ValueError("Concurrency must be >= 1")
    if Path(source_file).stem == "test" and not confirm_final_test:
        raise ValueError("Final test evaluation requires --confirm-final-test")
    if provider in {"teacher", "router"} and not execute_paid:
        raise ValueError("Teacher or router evaluation needs --execute-paid")
    # For paid evaluation, estimate before parallel calls; conservative full batch
    # upper bound cannot be guaranteed without provider-side spending limits.
    if provider in {"teacher", "router"} and budget_usd <= 0:
        raise ValueError("Paid evaluations require --budget-usd > 0")
    labels = load_labels(labels_file)
    rows = read_jsonl(source_file)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >=1")
        rows = rows[:limit]
    if not rows:
        raise ValueError("No evaluation rows")
    if provider == "student":
        predictor = VLLMStudent(labels, model, base_url)
    elif provider == "teacher":
        predictor = ClaudeTeacher(labels, model)
    elif provider == "router":
        import os
        predictor = InferenceRouter(
            VLLMStudent(labels, os.getenv("VLLM_MODEL", "banking77-lora"), base_url),
            ClaudeTeacher(labels, model), labels, threshold,
        )
    else:
        raise ValueError("provider must be student, teacher or router")

    def score_one(row):
        started = perf_counter()
        try:
            result = predictor.predict(row["text"])
            if provider == "router":
                prediction = result.prediction
                route, escalated = result.route, result.escalated
                latency_ms = result.total_latency_ms
                reason = result.reason
            else:
                prediction = result
                route, escalated = provider, False
                latency_ms = prediction.latency_ms
                reason = None
            return {
                "id": row["id"], "gold_intent": row["gold_intent"],
                "intent": prediction.intent, "status": "ok", "provider": provider,
                "model": prediction.model, "latency_ms": latency_ms,
                "input_tokens": prediction.input_tokens,
                "output_tokens": prediction.output_tokens,
                "confidence_proxy": prediction.confidence,
                "route": route, "escalated": escalated, "reason": reason,
            }
        except Exception as exc:
            return {
                "id": row["id"], "gold_intent": row["gold_intent"], "intent": None,
                "provider": provider, "model": model, "status": "failed",
                "latency_ms": (perf_counter() - started) * 1000,
                "input_tokens": None, "output_tokens": None,
                "escalated": False, "error_type": type(exc).__name__,
            }

    # Budget guard: paid evaluation runs one-at-a-time so spend is inspected
    # after every result. Concurrency benchmark uses --limit + provider budget
    # managed externally; it cannot hard-stop in-flight calls.
    if provider in {"teacher", "router"} and concurrency > 1:
        raise ValueError("Paid evaluation uses concurrency=1 to enforce local spend checks")
    outputs = []
    spent = 0.0
    if provider == "student":
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            outputs = list(executor.map(score_one, rows))
    else:
        for row in rows:
            if spent >= budget_usd:
                break
            result = score_one(row)
            outputs.append(result)
            if result["status"] == "ok" and (provider == "teacher" or result["escalated"]):
                spent += teacher_api_cost(
                    result.get("input_tokens") or 0, result.get("output_tokens") or 0,
                    input_price, output_price,
                )
    write_jsonl(output_file, outputs)
    return {
        "provider": provider, "concurrency": concurrency, "evaluated": len(outputs),
        "requested": len(rows), "estimated_paid_usd": round(spent, 6),
        "summary": score_predictions(outputs, labels),
        "warning": "Partial evaluation is not a full-test benchmark; API budget is approximate.",
    }
