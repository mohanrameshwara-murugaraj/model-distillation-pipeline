"""Inference FastAPI app; /classify is live only with configured model endpoints."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .io import append_jsonl
from .providers import ClaudeTeacher, VLLMStudent
from .router import InferenceRouter
from .taxonomy import load_labels


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)


class ClassifyResponse(BaseModel):
    intent: str
    route: str
    escalated: bool
    reason: str | None
    confidence_proxy: float | None
    latency_ms: float


def create_app(router: InferenceRouter | None = None, log_path: str | None = None) -> FastAPI:
    api = FastAPI(title="BANKING77 Distilled Intent Router", version="0.1.0")

    if router is None:
        # Provider objects are initialized lazily so health checks work without keys.
        @lru_cache(maxsize=1)
        def resolve_router() -> InferenceRouter:
            labels_path = os.getenv("LABELS_PATH", "data/processed/labels.json")
            labels = load_labels(labels_path)
            raw_threshold = os.getenv("ROUTER_MIN_CONFIDENCE", "")
            threshold = float(raw_threshold) if raw_threshold.strip() else None
            return InferenceRouter(
                VLLMStudent(
                    labels, os.getenv("VLLM_MODEL", "banking77-lora"),
                    os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                    os.getenv("VLLM_API_KEY", "EMPTY"),
                ),
                ClaudeTeacher(labels, os.getenv("TEACHER_MODEL", "claude-sonnet-4-6")),
                labels, min_confidence=threshold,
            )
    else:
        resolve_router = lambda: router

    @api.get("/health")
    def health():
        return {"status": "ok", "service": "distilled-intent-router"}

    @api.post("/classify", response_model=ClassifyResponse)
    def classify(payload: ClassifyRequest):
        try:
            routed = resolve_router().predict(payload.text)
        except Exception as exc:
            # Do not expose API keys, model exceptions or customer content.
            raise HTTPException(status_code=503, detail="Inference providers unavailable") from exc
        location = log_path if log_path is not None else os.getenv("ROUTER_LOG_PATH")
        if location:
            try:
                append_jsonl(Path(location), routed.safe_event())
            except OSError:
                # Telemetry failure shouldn't break inference.
                pass
        return ClassifyResponse(
            intent=routed.prediction.intent, route=routed.route, escalated=routed.escalated,
            reason=routed.reason, confidence_proxy=routed.prediction.confidence,
            latency_ms=round(routed.total_latency_ms, 2),
        )
    return api


load_dotenv()
app = create_app()
