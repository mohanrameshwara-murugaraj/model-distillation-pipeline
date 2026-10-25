"""Quality-gated teacher SFT. Optional gold-match ablation; validation uses gold."""
import json
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .taxonomy import STUDENT_SYSTEM, load_labels


def make_example(text: str, intent: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": STUDENT_SYSTEM},
            {"role": "user", "content": text},
        ],
        "completion": [{"role": "assistant", "content": intent}],
    }


def build(
    train_file: str, validation_file: str, labels_file: str,
    teacher_file: str, out_dir: str, require_gold_agreement: bool = False,
) -> dict:
    labels = set(load_labels(labels_file))
    train = read_jsonl(train_file)
    validation = read_jsonl(validation_file)
    teacher_rows = read_jsonl(teacher_file)
    indexed = {r["id"]: r for r in teacher_rows}
    if len(indexed) != len(teacher_rows):
        raise ValueError("Duplicate teacher IDs; fix/resume logs first")
    if not train:
        raise ValueError("Empty training set")
    val_ids = {r["id"] for r in validation}
    candidates = []
    stats = {"total_train_available": len(train), "teacher_labeled": 0,
             "teacher_failed_or_invalid": 0, "teacher_wrong_vs_gold": 0,
             "teacher_accepted": 0}
    for row in train:
        result = indexed.get(row["id"])
        if result is None:
            continue
        stats["teacher_labeled"] += 1
        predicted = result.get("intent")
        if result.get("status") != "ok" or predicted not in labels:
            stats["teacher_failed_or_invalid"] += 1
            continue
        if predicted != row["gold_intent"]:
            stats["teacher_wrong_vs_gold"] += 1
            if require_gold_agreement:
                continue
        if row["id"] in val_ids:
            raise ValueError("Train/validation leakage")
        candidates.append(make_example(row["text"], predicted))
    stats["teacher_accepted"] = len(candidates)
    stats["require_gold_agreement"] = require_gold_agreement
    if not candidates:
        raise ValueError("No accepted teacher examples. Run teacher labeling first.")
    dev_examples = [make_example(r["text"], r["gold_intent"]) for r in validation]
    folder = Path(out_dir)
    write_jsonl(folder / "sft_train.jsonl", candidates)
    write_jsonl(folder / "sft_validation.jsonl", dev_examples)
    (folder / "filter_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def build_gold_baseline(
    train_file: str, validation_file: str, labels_file: str,
    out_dir: str, match_teacher_file: str | None = None,
) -> dict:
    """Identical-task supervised baseline, matching teacher-labeled IDs if requested."""
    labels = set(load_labels(labels_file))
    train = read_jsonl(train_file)
    validation = read_jsonl(validation_file)
    selected_ids = None
    if match_teacher_file:
        teacher_rows = read_jsonl(match_teacher_file)
        selected_ids = {r["id"] for r in teacher_rows
                        if r.get("status") == "ok" and r.get("intent") in labels}
    selected = [r for r in train if selected_ids is None or r["id"] in selected_ids]
    if not selected:
        raise ValueError("No examples selected for gold baseline")
    if {r["id"] for r in selected} & {r["id"] for r in validation}:
        raise ValueError("Train/validation leakage")
    if any(r["gold_intent"] not in labels for r in selected + validation):
        raise ValueError("Invalid gold label")
    folder = Path(out_dir)
    write_jsonl(folder / "sft_train.jsonl",
                (make_example(r["text"], r["gold_intent"]) for r in selected))
    write_jsonl(folder / "sft_validation.jsonl",
                (make_example(r["text"], r["gold_intent"]) for r in validation))
    report = {"gold_training_count": len(selected),
              "match_teacher_file": match_teacher_file,
              "note": "Gold-supervision control. Train with same rank/epochs for comparison."}
    (folder / "baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
