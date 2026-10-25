"""Task metrics count malformed and failed predictions as incorrect."""


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def score_predictions(rows: list[dict], labels: list[str]) -> dict:
    if not rows:
        raise ValueError("No predictions")
    labelset = set(labels)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction ids")
    for row in rows:
        if row["gold_intent"] not in labelset:
            raise ValueError("Unknown gold label")
    actual = [row["gold_intent"] for row in rows]
    predicted = [row.get("intent") for row in rows]
    n = len(rows)
    accuracy = sum(a == p for a, p in zip(actual, predicted)) / n
    # Macro-F1 across gold-supported classes, including missing/invalid predictions.
    present = sorted(set(actual))
    f1s = []
    for label in present:
        tp = sum(a == p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0)
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") is not None]
    invalid = sum(p not in labelset for p in predicted)
    escalated = sum(bool(r.get("escalated")) for r in rows)
    input_tokens = sum(int(r.get("input_tokens") or 0) for r in rows)
    output_tokens = sum(int(r.get("output_tokens") or 0) for r in rows)
    token_covered = sum(
        r.get("input_tokens") is not None and r.get("output_tokens") is not None
        for r in rows
    )
    return {
        "samples": n,
        "accuracy": round(accuracy, 5), "macro_f1": round(sum(f1s) / len(f1s), 5),
        "classes_with_gold_support": len(present),
        "invalid_or_failed_outputs": invalid,
        "p50_latency_ms": round(percentile(latencies, 50), 3) if latencies else None,
        "p95_latency_ms": round(percentile(latencies, 95), 3) if latencies else None,
        "latency_samples": len(latencies),
        "escalation_rate": round(escalated / n, 5),
        "usage_covered": token_covered,
        "total_input_tokens": input_tokens, "total_output_tokens": output_tokens,
        "mean_input_tokens": round(input_tokens / token_covered, 2) if token_covered else None,
        "mean_output_tokens": round(output_tokens / token_covered, 2) if token_covered else None,
    }
