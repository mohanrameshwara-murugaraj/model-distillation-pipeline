
"""Free demonstration of student inference and teacher fallback."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

# Support execution directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_lab.providers import Prediction
from distill_lab.router import InferenceRouter
from distill_lab.taxonomy import load_labels


class MockPredictor:
    def __init__(self, intent, name):
        self.intent = intent
        self.name = name

    def predict(self, text):
        started = perf_counter()
        return Prediction(
            intent=self.intent,
            latency_ms=(perf_counter() - started) * 1000,
            model=self.name,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--mode", choices=["mock", "local"], default="mock"
    )
    parser.add_argument(
        "--simulate-invalid", action="store_true"
    )
    parser.add_argument(
        "--mock-intent", default="lost_or_stolen_card"
    )
    args = parser.parse_args()

    labels = load_labels("data/processed/labels.json")

    if args.mock_intent not in labels:
        parser.error("--mock-intent must be an official BANKING77 label")

    if args.mode == "local":
        from distill_lab.local_student import LocalQwenStudent
        student = LocalQwenStudent()
    else:
        student_intent = (
            "__invalid_intent__"
            if args.simulate_invalid
            else args.mock_intent
        )
        student = MockPredictor(student_intent, "mock-student")

    # This is a simulated response, NOT a Claude prediction.
    teacher = MockPredictor(args.mock_intent, "mock-teacher")

    router = InferenceRouter(
        student=student,
        teacher=teacher,
        labels=labels,
    )

    result = router.predict(args.text)

    print(json.dumps({
        "customer_message": args.text,
        "predicted_intent": result.prediction.intent,
        "route": result.route,
        "fallback_used": result.escalated,
        "fallback_reason": result.reason,
        "model": result.prediction.model,
        "teacher_mode": "simulated",
        "total_latency_ms": round(result.total_latency_ms, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
