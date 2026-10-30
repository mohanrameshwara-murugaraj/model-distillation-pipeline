# Reproducible experiment methodology

## Objective and scope

This repository explores the engineering path from a costly teacher API to a smaller banking-intent student model, with explicit routing, measurable quality, and break-even economics. **The completed experiment is a gold-label supervised QLoRA baseline, not teacher-to-student knowledge distillation.** The Claude teacher integration and paid-evaluation safeguards exist, but live teacher labeling could not be completed without API credits.

The immediate objective is an auditable personal portfolio prototype, not a production banking service or a demonstrated reduction in provider spending.

## Dataset and splits

The task is 77-way intent classification using PolyAI's BANKING77 (10,003 official training examples and 3,080 official test examples; CC BY 4.0). Intent names are read from the dataset metadata, not manually reordered. The official test split was not used for final accuracy reporting or model selection.

The repository's `distill prepare` path uses a stratified validation split and train-set duplicate checks. Its previously recorded manifest described 8,959 training examples and 1,001 validation examples after deduplication. **The completed Kaggle run used a separate lightweight sampling path**: a stratified 90/10 split of the original training portion, then up to 25 unique training examples and 5 validation examples per intent. Sampling used seed 42 and excluded exact normalized text matches with held-out samples; it did not reproduce the repository's character-shingle near-duplicate filtering. The intended train sample size was 1,925; the evaluation actually covered 382 validation examples. Confirm the exact train count from the saved Kaggle notebook before reporting it as independently verified.

The Kaggle sampling code also consulted official test texts only to exclude exact training or validation duplicates; it did **not** use their intent labels or test performance for selection. This differs from the repository's original documented test-isolation protocol and must be disclosed when comparing the two workflows. The results below are **validation metrics**, not official test metrics.

## Completed student experiment

- **Base model:** `Qwen/Qwen3-1.7B`.
- **Training data:** original BANKING77 human-provided intent labels (not Claude labels).
- **Environment:** Kaggle free NVIDIA Tesla T4 GPU, one visible GPU.
- **Method:** 4-bit NF4 QLoRA using Transformers, TRL `SFTTrainer`, and PEFT; completion-only loss.
- **Adapter:** LoRA rank 4, alpha 8, dropout 0.05, target `all-linear`.
- **Schedule:** one epoch, batch size 1, gradient accumulation 8, learning rate `1e-4`, maximum sequence length 192, seed 42.
- **Precision:** FP16 compute and FP32 trainable LoRA parameters. Non-reentrant gradient checkpointing was used to resolve compatibility issues encountered during initial tests.
- **Selection:** validation every 100 optimizer steps; best checkpoint selected by validation loss at step 200.
- **Outcome:** 241 optimizer steps; training progress bar reported 19 minutes 47 seconds; final reported training loss 0.7478059915; best validation loss 0.2161422074.

The adapter was exported as `qwen3-banking77-light-final.zip` and kept outside Git tracking. It requires the separate Qwen3-1.7B base model to perform real inference. The Kaggle training notebook should be committed separately (without credentials, datasets, or model weights) to make the actual experiment reproducible. The repository's original `scripts/train.py` describes a different configuration and is not evidence that it produced this adapter.

## Student validation and inference

Open-ended non-thinking generation on 382 development/validation examples gave:

| Metric | Measured value |
| --- | ---: |
| Accuracy | 0.5078534031 (50.79%) |
| Macro-F1 | 0.5371931820 |
| Non-taxonomy outputs | 100 / 382 |

Invalid labels were counted as incorrect, not discarded. Some outputs were semantically plausible but did not exactly match the BANKING77 taxonomy (for example, `delete_account` instead of `terminate_account`). Increasing generation length from 32 to 64 tokens did not correct the inspected examples.

An exploratory candidate-label scoring experiment chose among the 77 official labels and corrected 5 out of 10 *deliberately selected previously invalid* cases. This subset is not an unbiased comparison, and a complete closed-set validation benchmark was not reported. Do not present it as a measured overall improvement.

The training notebook's system prompt differs from the repository's original `STUDENT_SYSTEM`; the optional local adapter provider reproduces the training prompt. This interface mismatch should be resolved and checked before claiming reproducible production inference quality. Real local model inference and GPU-hosted throughput have not been verified as part of the portfolio demonstration.

## Teacher integration and fallback: implemented versus measured

The existing `ClaudeTeacher` provider requests a schema-constrained label, and teacher collection requires explicit paid execution and a budget argument. The live Claude pilot was blocked by insufficient API credits. **No successful teacher-labeled training corpus, teacher quality benchmark, or teacher-cost measurement has been produced.**

The existing `InferenceRouter` checks student output labels and supports fallback on invalid predictions or provider exceptions; a confidence-proxy threshold is optional and not calibrated. The demo script exercises both student and teacher branches with fixed **mock** predictions. These tests verify routing behavior only; mock outputs are not Claude predictions and mock timing is not model latency. The project had 17 passing offline tests after integration.

The pre-existing vLLM provider and FastAPI service are implementation paths, not evidence of deployed serving throughput. The vLLM launch script must be configured for the actual 1.7B/rank-4 adapter before a real GPU serving test.

## Benchmark and cost methodology

`benchmark.py` requires comparable prediction records with the same example IDs and gold labels. Since no real Claude evaluation exists, the repository cannot yet claim a teacher-versus-student latency, accuracy, or cost benchmark. Do not construct fabricated teacher predictions to populate that table.

The `EconomicsInputs` module computes an **illustrative** always-on GPU scenario, not real spend. A saved scenario assumes teacher cost $0.003/request, GPU $0.50/hour, operations $1/day, student variable cost $0, and 25% fallback. It yields a modeled break-even of 5,778 requests/day. At 10,000 requests/day, hypothetical teacher-only cost is $30.00/day versus $20.50/day student plus fallback, a modeled $9.50/day difference. Neither teacher token usage nor a sustainable requests-per-second capacity was measured; feasibility under a latency service level remains unknown.

The economic model includes continuously billed GPU idle time and fallback costs. Capacity-aware conclusions require a measured serving load test, matched quality, actual API usage, and stated rates.

## Reproduction and limitations

- **No GPU required locally:** `python -m pytest -q` and `python scripts/run_demo.py --text "I lost my bank card"` exercise the offline mock path. `--simulate-invalid` demonstrates fallback.
- **GPU needed for real model evaluation:** reload the Qwen3-1.7B base model and exported LoRA adapter in Kaggle or another compatible inference environment; do not confuse a mock run with real student predictions.
- **Data reproducibility:** keep the exact lightweight Kaggle sampling procedure and seed with the notebook. Do not describe results from that sampling path as outputs of the full repository dedup pipeline.
- **Evaluation:** the reported development metrics are not a once-only official test estimate; no production dataset or real customer traffic was evaluated.
- **Confidence:** token-generation confidence proxies are not calibrated classification probabilities; threshold-based routing requires validation.
- **Domain and privacy:** fine-grained labels can be ambiguous; real customer messages may contain sensitive personal data and must not be sent to an external provider without appropriate safeguards.

## Future work (not completed)

1. Obtain permission and credits for an explicitly budgeted Claude teacher pilot, then compare teacher-labeled SFT against a matched human-gold baseline.
2. Run an actual Qwen adapter through the repository's inference serving path, measure end-to-end latency and throughput, and calibrate a fallback policy.
3. Perform a final locked-test evaluation and a capacity-constrained economics comparison using real usage and pricing data.
