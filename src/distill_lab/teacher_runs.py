"""Paid teacher labeling is opt-in, resumable, and logs actual usage."""
from pathlib import Path

from .economics import teacher_api_cost
from .io import append_jsonl, read_jsonl
from .providers import ClaudeTeacher
from .taxonomy import load_labels


def run_teacher(
    *, input_file: str, output_file: str, labels_file: str, model: str,
    limit: int, execute_paid: bool, budget_usd: float,
    input_price: float, output_price: float,
) -> dict:
    if not execute_paid:
        raise ValueError("Teacher API calls are disabled: pass --execute-paid explicitly")
    if budget_usd <= 0 or limit <= 0:
        raise ValueError("Positive --budget-usd and --limit are required")
    if input_price < 0 or output_price < 0:
        raise ValueError("Pricing must be nonnegative")
    source = read_jsonl(input_file)
    labels = load_labels(labels_file)
    prior = read_jsonl(output_file) if Path(output_file).exists() else []
    existing_ids = {r["id"] for r in prior}
    spent = 0.0
    for r in prior:
        if r.get("model") == model:
            spent += float(r.get("estimated_usd") or 0)
    teacher = ClaudeTeacher(labels, model)
    attempted = 0
    success = 0
    for row in source:
        if attempted >= limit or spent >= budget_usd:
            break
        if row["id"] in existing_ids:
            continue
        attempted += 1
        try:
            prediction = teacher.predict(row["text"])
            cost = teacher_api_cost(
                prediction.input_tokens or 0, prediction.output_tokens or 0,
                input_price, output_price,
            )
            record = {
                "id": row["id"], "gold_intent": row["gold_intent"],
                "intent": prediction.intent, "model": model,
                "latency_ms": prediction.latency_ms,
                "input_tokens": prediction.input_tokens,
                "output_tokens": prediction.output_tokens,
                "estimated_usd": round(cost, 9), "status": "ok",
            }
            success += 1
            spent += cost
        except Exception as exc:
            # API exception messages can contain request data; do not persist them.
            record = {
                "id": row["id"], "gold_intent": row["gold_intent"],
                "intent": None, "status": "failed", "model": model,
                "latency_ms": None, "input_tokens": None, "output_tokens": None,
                "estimated_usd": None,
                "error_type": type(exc).__name__,
            }
        append_jsonl(output_file, record)
        existing_ids.add(row["id"])
    return {
        "new_attempts": attempted, "new_successes": success,
        "estimated_cumulative_usd_in_output": round(spent, 6),
        "budget_usd": budget_usd,
        "caution": "Budget is checked between calls and may be exceeded by one request; pricing excludes cache effects/taxes.",
    }
