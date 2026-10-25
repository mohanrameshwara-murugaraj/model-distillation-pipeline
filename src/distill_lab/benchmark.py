"""Truthful benchmark: identical example IDs, explicit sample coverage and failure counts."""
import json
from pathlib import Path

from .io import read_jsonl
from .metrics import score_predictions
from .taxonomy import load_labels


def benchmark(teacher_path: str, student_path: str, labels_path: str,
              output_dir: str, expected: int | None = None,
              router_path: str | None = None) -> dict:
    labels = load_labels(labels_path)
    paths = {"Teacher": teacher_path, "Student": student_path}
    if router_path:
        paths["Router"] = router_path
    sources = {name: read_jsonl(path) for name, path in paths.items()}
    ids = {name: {r["id"] for r in rows} for name, rows in sources.items()}
    if any(len(sources[name]) != len(idset) for name, idset in ids.items()):
        raise ValueError("Duplicate IDs in one or more files")
    baseline = ids["Teacher"]
    if any(idset != baseline for idset in ids.values()):
        raise ValueError("Benchmark requires identical sample IDs for all providers")
    teacher_golds = {r["id"]: r["gold_intent"] for r in sources["Teacher"]}
    for name, rows in sources.items():
        if any(teacher_golds[r["id"]] != r["gold_intent"] for r in rows):
            raise ValueError(f"Gold-label mismatch between Teacher and {name}")
    if expected is not None and len(baseline) != expected:
        raise ValueError(f"Expected {expected} examples, found {len(baseline)}")
    summaries = {name: score_predictions(rows, labels) for name, rows in sources.items()}
    parts = [
        "# Evaluation results",
        "",
        f"Evaluation set size: **{len(baseline)}** identical examples per provider.",
        "",
        "| System | Accuracy | Macro-F1 | p50 latency | p95 latency | Invalid/failed | Escalation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, v in summaries.items():
        def ms(key):
            return f"{v[key]:.1f} ms" if v[key] is not None else "N/A"
        parts.append(
            f"| {name} | {v['accuracy']:.2%} | {v['macro_f1']:.3f} | "
            f"{ms('p50_latency_ms')} | {ms('p95_latency_ms')} | "
            f"{v['invalid_or_failed_outputs']} | {v['escalation_rate']:.1%} |"
        )
    parts += [
        "", "**Important:** Quality measured against original BANKING77 gold labels, not teacher labels.",
        "Latency is client-side wall clock. Compare only runs under equivalent hardware, "
        "concurrency, region, connection reuse and warmup conditions.",
        "Costs are produced separately from measured usage and stated GPU pricing; "
        "do not present an unmeasured estimate as a benchmark.",
    ]
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "benchmark.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    result = {"sample_count": len(baseline), "summaries": summaries}
    (folder / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
