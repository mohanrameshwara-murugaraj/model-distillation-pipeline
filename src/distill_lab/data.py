"""Prepare real BANKING77 data with frozen official test and deterministic validation."""
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from .io import write_jsonl


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", text.casefold())).strip()


def example_id(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:24]


def fingerprint(text: str) -> set[str]:
    s = f"  {normalize(text)}  "
    return {s[i : i + 5] for i in range(max(1, len(s) - 4))}


def dedup_train(
    train: list[dict], reserved: list[dict], *, threshold: float = 0.90
) -> tuple[list[dict], dict[str, int]]:
    """Remove exact/near train duplicates and train rows too similar to held-out rows.

    Only training is modified. Using a shingle inverted index avoids global O(n^2)
    pairwise comparisons; identical short strings are caught by exact matching.
    """
    postings: dict[str, list[int]] = defaultdict(list)
    indexed: list[set[str]] = []
    seen: set[str] = set()
    for item in reserved:
        key = normalize(item["text"])
        if key in seen:
            continue
        seen.add(key)
        grams = fingerprint(item["text"])
        j = len(indexed)
        indexed.append(grams)
        for gram in grams:
            postings[gram].append(j)
    kept: list[dict] = []
    removed_exact = removed_near = 0
    for item in train:
        key = normalize(item["text"])
        if key in seen:
            removed_exact += 1
            continue
        grams = fingerprint(item["text"])
        candidate_overlap: Counter[int] = Counter()
        for gram in grams:
            candidate_overlap.update(postings.get(gram, []))
        similar = False
        for candidate, overlap in candidate_overlap.items():
            union = len(grams) + len(indexed[candidate]) - overlap
            if union and overlap / union >= threshold:
                similar = True
                break
        if similar:
            removed_near += 1
            continue
        seen.add(key)
        j = len(indexed)
        indexed.append(grams)
        for gram in grams:
            postings[gram].append(j)
        kept.append(item)
    return kept, {"removed_exact": removed_exact, "removed_near": removed_near}


def prepare(out: str = "data/processed", seed: int = 42, valid_fraction: float = 0.10) -> dict:
    from datasets import load_dataset  # deferred: simple API/test operation does not need HF

    if not 0.05 <= valid_fraction <= 0.3:
        raise ValueError("valid_fraction must be in [0.05, 0.30]")
    ds = load_dataset("PolyAI/banking77")
    labels = ds["train"].features["label"].names
    # The official test split never participates in selection, training or de-duplication.
    split = ds["train"].train_test_split(
        test_size=valid_fraction, stratify_by_column="label", seed=seed
    )

    def records(part):
        return [
            {"id": example_id(row["text"]), "text": row["text"],
             "gold_intent": labels[int(row["label"])]}
            for row in part
        ]

    train, validation, test = (
        records(split["train"]), records(split["test"]), records(ds["test"])
    )
    train, stats = dedup_train(train, validation)
    folder = Path(out)
    for name, items in [("train", train), ("validation", validation), ("test", test)]:
        write_jsonl(folder / f"{name}.jsonl", items)
    import json
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    manifest = {
        "dataset": "PolyAI/banking77", "seed": seed, "validation_fraction": valid_fraction,
        "train_count_after_dedup": len(train), "validation_count": len(validation),
        "official_test_count": len(test), "labels": len(labels), **stats,
        "warning": "Original test split kept unchanged; never tune on test."
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
