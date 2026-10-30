"""Real Claude teacher and vLLM student adapters (no network calls on import)."""
import json
import math
import time
from dataclasses import dataclass
from typing import Protocol

from .taxonomy import STUDENT_SYSTEM, assert_valid_label, teacher_system


@dataclass(frozen=True)
class Prediction:
    intent: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    confidence: float | None = None
    model: str = ""


class Predictor(Protocol):
    def predict(self, text: str) -> Prediction: ...


class ClaudeTeacher:
    def __init__(self, labels: list[str], model: str, client=None):
        self.labels = labels
        self.model = model
        if client is None:
            from anthropic import Anthropic
            client = Anthropic(timeout=60.0, max_retries=2)
        self.client = client

    def predict(self, text: str) -> Prediction:
        # A schema-constrained response still needs semantic label validation.
        schema = {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": self.labels}},
            "required": ["intent"], "additionalProperties": False,
        }
        started = time.perf_counter()
        result = self.client.messages.create(
            model=self.model,
            max_tokens=128,
            system=teacher_system(self.labels),
            messages=[{"role": "user", "content": text}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        elapsed = (time.perf_counter() - started) * 1000
        blocks = [block.text for block in result.content if block.type == "text"]
        if not blocks:
            raise ValueError("Teacher produced no text block")
        parsed = json.loads("".join(blocks))
        intent = assert_valid_label(parsed.get("intent"), self.labels)
        return Prediction(
            intent=intent, latency_ms=elapsed, model=self.model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )


class VLLMStudent:
    """vLLM OpenAI-compatible endpoint with constrained intent choices.

    Logprob-based proxy is NOT calibrated class probability. Gate threshold must
    be selected on validation data. Missing logprobs produce confidence=None.
    """

    def __init__(self, labels: list[str], model: str, base_url: str, api_key: str = "EMPTY", client=None):
        self.labels = labels
        self.model = model
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=1)
        self.client = client

    def predict(self, text: str) -> Prediction:
        started = time.perf_counter()
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": STUDENT_SYSTEM},
                      {"role": "user", "content": text}],
            max_tokens=80,
            logprobs=True,
            extra_body={
                "structured_outputs": {"choice": self.labels},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        elapsed = (time.perf_counter() - started) * 1000
        choice = result.choices[0]
        intent = assert_valid_label((choice.message.content or "").strip(), self.labels)
        token_logprobs = getattr(getattr(choice, "logprobs", None), "content", None)
        confidence = None
        if token_logprobs:
            scores = [v.logprob for v in token_logprobs if v.logprob is not None]
            if scores:
                # Geometric mean token likelihood; NOT posterior P(class | query).
                confidence = max(0.0, min(1.0, math.exp(sum(scores) / len(scores))))
        usage = getattr(result, "usage", None)
        return Prediction(
            intent=intent, latency_ms=elapsed, confidence=confidence,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None), model=self.model,
        )
