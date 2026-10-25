import pytest

from distill_lab.economics import EconomicsInputs, teacher_api_cost


def test_api_token_cost():
    assert teacher_api_cost(1000, 100, 3.0, 15.0) == pytest.approx(0.0045)


def test_idle_cost_escalation_and_break_even():
    inputs = EconomicsInputs(
        teacher_usd_per_request=0.01, gpu_usd_per_hour=1,
        escalation_rate=0.1, operations_usd_per_day=0,
    )
    assert inputs.fixed_daily_usd == 24
    assert inputs.student_marginal_usd == pytest.approx(.001)
    assert inputs.break_even_requests_per_day() == 2667
    assert inputs.cost_at_volume(3000)["student_router_daily_usd"] == 27
    assert inputs.cost_at_volume(3000)["teacher_daily_usd"] == 30


def test_no_break_even_when_fallback_or_variable_cost_too_high():
    inputs = EconomicsInputs(teacher_usd_per_request=0.002, gpu_usd_per_hour=1,
                             escalation_rate=1)
    assert inputs.break_even_requests_per_day() is None
    assert inputs.as_report(100)["break_even_within_measured_capacity"] is None


def test_capacity_can_make_break_even_infeasible():
    inputs = EconomicsInputs(teacher_usd_per_request=0.01, gpu_usd_per_hour=1,
                             escalation_rate=0, acceptable_rps=0.01)
    assert inputs.as_report(5000)["break_even_within_measured_capacity"] is False


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        EconomicsInputs(0.01, 1, 1.01)
    with pytest.raises(ValueError):
        EconomicsInputs(0.01, -1, 0)
