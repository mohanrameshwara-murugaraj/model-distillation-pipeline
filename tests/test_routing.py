from types import SimpleNamespace

from fastapi.testclient import TestClient

from distill_lab.api import create_app
from distill_lab.providers import ClaudeTeacher, Prediction, VLLMStudent
from distill_lab.router import InferenceRouter


class Fake:
    def __init__(self, result=None, fail=None):
        self.result = result
        self.fail = fail
        self.calls = 0

    def predict(self, text):
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.result


def pred(label, confidence=None):
    return Prediction(intent=label, latency_ms=2.0, confidence=confidence,
                      model="fixture")


def test_good_student_never_calls_paid_teacher():
    student = Fake(pred("card_arrival", .92))
    teacher = Fake(pred("card_arrival"))
    router = InferenceRouter(student, teacher, ["card_arrival", "change_pin"], .6)
    result = router.predict("When does my card arrive?")
    assert result.route == "student"
    assert not result.escalated
    assert teacher.calls == 0
    assert "text" not in result.safe_event()


def test_low_confidence_escalates():
    teacher = Fake(pred("card_arrival"))
    router = InferenceRouter(Fake(pred("change_pin", .20)), teacher,
                             ["card_arrival", "change_pin"], .6)
    result = router.predict("Where is my card?")
    assert result.route == "teacher"
    assert result.reason == "low_confidence"
    assert teacher.calls == 1


def test_invalid_and_error_escalate():
    for student in (Fake(pred("unlisted")), Fake(fail=ConnectionError("down"))):
        teacher = Fake(pred("card_arrival"))
        result = InferenceRouter(student, teacher, ["card_arrival"]).predict("Hi")
        assert result.escalated
        assert teacher.calls == 1


def test_endpoint_and_health_work_without_actual_services():
    router = InferenceRouter(Fake(pred("card_arrival")), Fake(pred("card_arrival")),
                             ["card_arrival"])
    client = TestClient(create_app(router=router, log_path=""))
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/classify", json={"text": "Card arrival?"})
    assert response.status_code == 200
    assert response.json()["route"] == "student"
    assert client.post("/classify", json={"text": ""}).status_code == 422


def test_teacher_schema_and_usage_capture():
    def teacher_request(**kwargs):
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"intent":"change_pin"}')],
            usage=SimpleNamespace(input_tokens=50, output_tokens=7),
        )
    stub = SimpleNamespace(messages=SimpleNamespace(create=teacher_request))
    result = ClaudeTeacher(["change_pin"], "fixture", client=stub).predict("PIN change")
    assert result.intent == "change_pin"
    assert result.input_tokens == 50


def test_student_constrained_choice_and_logprob():
    def student_request(**kwargs):
        assert kwargs["extra_body"]["structured_outputs"]["choice"] == ["change_pin"]
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="change_pin"),
                logprobs=SimpleNamespace(content=[SimpleNamespace(logprob=-.1)]),
            )],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=2),
        )
    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=student_request)))
    result = VLLMStudent(["change_pin"], "fixture", "http://localhost", client=stub).predict("PIN change")
    assert .8 < result.confidence < 1
