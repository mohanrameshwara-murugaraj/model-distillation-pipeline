"""Escalation on student exception, invalid intent, or a validation-calibrated threshold."""
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

from .providers import Prediction, Predictor
from .taxonomy import assert_valid_label


@dataclass(frozen=True)
class RoutedPrediction:
    prediction: Prediction
    route: str
    escalated: bool
    reason: str | None
    total_latency_ms: float
    student_latency_ms: float | None

    def safe_event(self) -> dict:
        """No customer text and no customer ID in the operational event."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": self.route,
            "escalated": self.escalated,
            "reason": self.reason,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "student_latency_ms": self.student_latency_ms,
            "final_model": self.prediction.model,
            "input_tokens": self.prediction.input_tokens,
            "output_tokens": self.prediction.output_tokens,
        }


class InferenceRouter:
    def __init__(
        self, student: Predictor, teacher: Predictor, labels: list[str],
        min_confidence: float | None = None,
    ):
        if min_confidence is not None and not (0 <= min_confidence <= 1):
            raise ValueError("min_confidence must be between zero and one")
        self.student = student
        self.teacher = teacher
        self.labels = labels
        self.min_confidence = min_confidence

    def predict(self, text: str) -> RoutedPrediction:
        started = perf_counter()
        student_latency = None
        reason = None
        try:
            student_prediction = self.student.predict(text)
            student_latency = student_prediction.latency_ms
            assert_valid_label(student_prediction.intent, self.labels)
            if self.min_confidence is not None:
                if student_prediction.confidence is None:
                    reason = "missing_confidence"
                elif student_prediction.confidence < self.min_confidence:
                    reason = "low_confidence"
            if reason is None:
                return RoutedPrediction(
                    student_prediction, "student", False, None,
                    (perf_counter() - started) * 1000, student_latency,
                )
        except (Exception) as exc:
            # Provider errors and invalid outputs both trigger fallback. API errors
            # never leak exception details to the outside caller.
            reason = "invalid_student_output" if isinstance(exc, ValueError) else "student_error"
        teacher_prediction = self.teacher.predict(text)
        assert_valid_label(teacher_prediction.intent, self.labels)
        return RoutedPrediction(
            teacher_prediction, "teacher", True, reason,
            (perf_counter() - started) * 1000, student_latency,
        )
