import json

import pytest

from distill_lab.benchmark import benchmark
from distill_lab.build_sft import build, build_gold_baseline
from distill_lab.io import read_jsonl, write_jsonl
from distill_lab.metrics import percentile, score_predictions


def test_quality_gate_and_validation(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["card_arrival", "change_pin"]))
    train, val, teacher, out = [tmp_path / x for x in
                                ("train.jsonl", "validation.jsonl", "teacher.jsonl", "out")]
    write_jsonl(train, [
        {"id": "a", "text": "Where is my card", "gold_intent": "card_arrival"},
        {"id": "b", "text": "PIN reset", "gold_intent": "change_pin"},
        {"id": "c", "text": "Another card", "gold_intent": "card_arrival"},
    ])
    write_jsonl(val, [{"id": "v", "text": "Help change PIN", "gold_intent": "change_pin"}])
    write_jsonl(teacher, [
        {"id": "a", "intent": "card_arrival", "status": "ok"},
        {"id": "b", "intent": "card_arrival", "status": "ok"},
        {"id": "c", "intent": "not_a_label", "status": "ok"},
    ])
    stats = build(str(train), str(val), str(labels), str(teacher), str(out))
    assert stats["teacher_accepted"] == 2
    assert stats["teacher_wrong_vs_gold"] == 1
    assert stats["teacher_failed_or_invalid"] == 1
    assert len(read_jsonl(out / "sft_validation.jsonl")) == 1
    strict = tmp_path / "gold-filtered"
    strict_stats = build(str(train), str(val), str(labels), str(teacher),
                         str(strict), require_gold_agreement=True)
    assert strict_stats["teacher_accepted"] == 1
    baseline = tmp_path / "baseline"
    baseline_stats = build_gold_baseline(str(train), str(val), str(labels),
                                        str(baseline), str(teacher))
    assert baseline_stats["gold_training_count"] == 2
    assert read_jsonl(out / "sft_train.jsonl")[0]["completion"][0]["content"] == "card_arrival"


def test_benchmark_refuses_mismatched_ids_and_makes_report(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["card_arrival", "change_pin"]))
    teacher = tmp_path / "teacher.jsonl"
    student = tmp_path / "student.jsonl"
    write_jsonl(teacher, [{"id": "a", "gold_intent": "card_arrival",
                           "intent": "card_arrival", "latency_ms": 10}])
    write_jsonl(student, [{"id": "b", "gold_intent": "change_pin",
                           "intent": "change_pin", "latency_ms": 15}])
    with pytest.raises(ValueError, match="identical"):
        benchmark(str(teacher), str(student), str(labels), str(tmp_path / "result"))
    write_jsonl(student, [{"id": "a", "gold_intent": "card_arrival",
                           "intent": None, "latency_ms": 15}])
    result = benchmark(str(teacher), str(student), str(labels),
                       str(tmp_path / "result"), expected=1)
    assert result["summaries"]["Student"]["accuracy"] == 0
    assert (tmp_path / "result" / "benchmark.md").exists()


def test_metric_invalid_is_incorrect():
    result = score_predictions([
        {"id": "a", "gold_intent": "a", "intent": None, "latency_ms": 100},
        {"id": "b", "gold_intent": "b", "intent": "b", "latency_ms": 200},
    ], ["a", "b"])
    assert result["accuracy"] == .5
    assert result["invalid_or_failed_outputs"] == 1
    assert percentile([100, 200], 95) == pytest.approx(195)
