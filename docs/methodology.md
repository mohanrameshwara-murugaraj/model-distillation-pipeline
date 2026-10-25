# Reproducible experiment methodology

## Objective and decision criterion

Determine whether a Claude-powered 77-way intent classification task can be served more economically by a fine-tuned Qwen3-8B adapter, with an honest quality/latency comparison, an accountable fallback rate, and a measured capacity constraint. **Cheaper is not better if quality or latency falls below the required service level.**

## Dataset provenance

BANKING77, PolyAI, 2020, CC BY 4.0. Official train: 10,003, official test: 3,080. All 77 fine-grained banking intents are retained. Code obtains the label order from dataset metadata rather than a manually sorted list. Original test data is frozen and never used for model selection, label filtering or threshold selection.

Using random seed 42, 10% of the original training partition becomes stratified validation. Identical and highly overlapping character-shingle training samples are removed where they collide with validation or earlier retained train samples. The official test partition is not consulted for selection or filtering. The held-out data is not modified. Log removal counts and inspect remaining leakage risk.

## Teacher pilot and filtering

Claude classifies real training/query inputs under a schema with exactly one enum intent. Benchmark Claude first on a validation pilot against original gold labels. For the chosen teacher, sample/label several thousand original training examples, record provider model ID, schema errors, latency and input/output token counts. For primary student supervision, accept schema-valid teacher predictions as delivered, including those disagreeing with gold. An optional gold-agreement-filtered ablation measures error-filtering trade-offs. Train a separate human-gold baseline with the same labeled query IDs. No teacher outputs from the test partition enter training.

To ensure transparent costs, input/output price assumptions are passed at execution, and individual measured token counts are persisted to ignored local JSONL. API failures are not quietly treated as correct outputs.

## Student model and hyperparameters

Baseline: Qwen/Qwen3-8B, adapter-only supervised fine-tuning using TRL SFTTrainer and PEFT, completion-only loss. Two planned runs: QLoRA rank 8 and rank 32, alpha = 2×rank, dropout = 0.05, learning rate = 1e-4, maximum 3 epochs, random seed 42, batch=2, gradient accumulation=8. Save and evaluate at each epoch. The training prompt matches the student serving instruction; generation is configured as non-thinking.

Use W&B for training/validation loss and checkpoint samples. Run `scripts/validate_checkpoints.py` for generated-label accuracy and sample predictions after each epoch; select the final adapter on validation only. Compare a non-tuned Qwen3-8B baseline too if resources allow; without it, attribute improvements cautiously.

## Evaluation protocol

Score both teacher and student against **gold test labels**, not teacher predictions. Accuracy, macro-F1, invalid/error rate, per-intent breakdown (future extension), latency p50/p95. Treat failed predictions as incorrect; do not drop them from the denominator. Compare only identical example IDs, and enforce `--expected 3080` on the official test.

Run student and teacher with identical concurrency for latency comparison. Local teacher benchmarking restricts concurrency to one for cost-guard purposes; measure student at concurrency one for the directly comparable row. For serving throughput, warm the GPU, sweep concurrency levels, measure all requests including fallbacks, and select the highest observed throughput meeting a stated p95 SLO. Avoid relying on short burst results as a sustained-capacity guarantee.

## Confidence and escalation

Choice-constrained vLLM output is always one of the allowed intent strings in engines supporting choice constraints; however, it cannot detect unfamiliar/out-of-distribution intents. A student output error or invalid label escalates to Claude. Optional token-logprob-based confidence gate uses the geometric mean of generated-token probabilities. This proxy is not inherently calibrated or comparable across tokenizations. Select a threshold on validation using a quality-coverage curve. If logits are unavailable and gating is enabled, escalate.

## Unit economics

Let T be observed teacher USD/request; H be real billed GPU USD/hour, F be non-GPU fixed daily costs and training/setup amortization per day, v be extra variable USD/student request, and e be measured fraction routed to teacher. Then API-only daily spend at N requests is NT; student+router spend is 24H + F + N(v + eT). A positive break-even only exists if T > v + eT, and it is practically feasible only if its load fits measured capacity while respecting p95 latency.

Use actual billed rates and observed traffic mix, state whether the GPU is on demand or reserved, include overhead and idle time, and keep training costs separate or amortize explicitly. Report scenario sensitivity across GPU rates, escalations, and traffic volumes. Never use a hypothetical example value as an achieved savings claim.

## Limitations and ethical considerations

Public dataset query distribution might diverge from a real production bank. Golden labels can contain ambiguity. Optional teacher/gold agreement filtering may bias training toward easier examples. The human-gold baseline could outperform distillation, in which case API labeling would have no incremental value on this already labeled dataset. Query texts may contain sensitive details in real operations: scrub, anonymize, and apply legal/contractual handling rules before invoking a paid external provider. No real customer PII needs to be used for this educational demonstration.
