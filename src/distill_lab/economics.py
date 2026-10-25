"""Explicit fully loaded always-on GPU unit economics, with fallback overhead."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class EconomicsInputs:
    teacher_usd_per_request: float
    gpu_usd_per_hour: float
    escalation_rate: float
    student_variable_usd_per_request: float = 0.0
    operations_usd_per_day: float = 0.0
    amortized_setup_usd_per_day: float = 0.0
    acceptable_rps: float | None = None

    def __post_init__(self):
        costs = [self.teacher_usd_per_request, self.gpu_usd_per_hour,
                 self.student_variable_usd_per_request, self.operations_usd_per_day,
                 self.amortized_setup_usd_per_day]
        if any(not isfinite(v) or v < 0 for v in costs):
            raise ValueError("Costs must be finite and nonnegative")
        if not (0 <= self.escalation_rate <= 1):
            raise ValueError("Escalation rate must be in [0, 1]")
        if self.acceptable_rps is not None and (
            not isfinite(self.acceptable_rps) or self.acceptable_rps <= 0
        ):
            raise ValueError("acceptable_rps must be positive")

    @property
    def fixed_daily_usd(self) -> float:
        return (24 * self.gpu_usd_per_hour + self.operations_usd_per_day
                + self.amortized_setup_usd_per_day)

    @property
    def student_marginal_usd(self) -> float:
        return (self.student_variable_usd_per_request
                + self.escalation_rate * self.teacher_usd_per_request)

    @property
    def daily_savings_per_request(self) -> float:
        return self.teacher_usd_per_request - self.student_marginal_usd

    def break_even_requests_per_day(self) -> int | None:
        """First integer volume with student+fallback cheaper (strictly below API)."""
        saving = self.daily_savings_per_request
        if saving <= 0:
            return None
        return int(self.fixed_daily_usd // saving) + 1

    def cost_at_volume(self, requests_per_day: int) -> dict:
        if requests_per_day <= 0:
            raise ValueError("requests_per_day must be positive")
        teacher = requests_per_day * self.teacher_usd_per_request
        student = self.fixed_daily_usd + requests_per_day * self.student_marginal_usd
        return {
            "requests_per_day": requests_per_day,
            "teacher_daily_usd": round(teacher, 6),
            "student_router_daily_usd": round(student, 6),
            "net_savings_daily_usd": round(teacher - student, 6),
            "teacher_usd_per_1000": round(1000 * self.teacher_usd_per_request, 6),
            "student_router_usd_per_1000": round(1000 * student / requests_per_day, 6),
        }

    def as_report(self, volume: int) -> dict:
        break_even = self.break_even_requests_per_day()
        daily_capacity = (
            int(86400 * self.acceptable_rps) if self.acceptable_rps is not None else None
        )
        return {
            "assumptions": vars(self), "fixed_daily_usd_including_idle": self.fixed_daily_usd,
            "student_marginal_usd_including_escalations": self.student_marginal_usd,
            "break_even_requests_per_day": break_even,
            "measured_slo_capacity_requests_per_day": daily_capacity,
            "break_even_within_measured_capacity": (
                break_even <= daily_capacity if break_even is not None
                and daily_capacity is not None else None
            ),
            "at_requested_volume": self.cost_at_volume(volume),
        }


def teacher_api_cost(
    input_tokens: int, output_tokens: int,
    usd_per_million_input_tokens: float, usd_per_million_output_tokens: float,
) -> float:
    if min(input_tokens, output_tokens, usd_per_million_input_tokens,
           usd_per_million_output_tokens) < 0:
        raise ValueError("Token counts and prices cannot be negative")
    return ((input_tokens * usd_per_million_input_tokens
             + output_tokens * usd_per_million_output_tokens) / 1_000_000)
