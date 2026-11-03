# Project Summary — LLM Fine-Tuning & Inference Routing Pipeline

## Project Overview

This project explores how a smaller language model can be adapted for a specialized banking customer-support task, with a modular architecture for teacher–student learning, inference routing, and deployment cost analysis.

The objective was to build a reproducible AI engineering workflow using accessible tools and free GPU resources, rather than develop a production banking application.

## What I Built

**1. Data Preparation**

Used the BANKING77 dataset containing 77 banking support intents. Implemented data preparation, stratified validation splitting, duplicate checks, and label validation.

**2. Student Model Fine-Tuning**

Fine-tuned Qwen3-1.7B using Hugging Face Transformers, TRL, PEFT, and 4-bit QLoRA on a free Kaggle Tesla T4 GPU.

Used human-labeled training examples, a lightweight single-epoch configuration, and LoRA adapters to reduce training resource requirements.

**3. Model Evaluation**

Evaluated the fine-tuned model against unseen development-validation examples, measuring classification accuracy, macro-F1, and invalid-label predictions.

**4. Inference Routing and Fallback**

Implemented an inference architecture that validates student predictions and supports escalation to a teacher model when the student produces invalid outputs or encounters errors.

Demonstrated the routing workflow with simulated student and teacher providers, without requiring paid API access.

**5. Deployment Economics**

Implemented a cost model comparing API-based inference with self-hosted student inference and optional teacher fallback. Explored break-even traffic volumes under explicitly stated hypothetical assumptions.

## Experimental Results

| Metric | Result |
|---|---|
| Base model | Qwen3-1.7B |
| Fine-tuning method | 4-bit QLoRA |
| LoRA rank | 4 |
| Training epochs | 1 |
| Training runtime | 19 minutes 47 seconds |
| Validation examples | 382 |
| Validation accuracy | 50.79% |
| Macro-F1 | 0.5372 |
| Automated tests | 17 passed |

Training and validation loss, prediction outcomes, and hypothetical deployment economics are visualized in the main README.

## Technologies

Python · PyTorch · Hugging Face Transformers · Datasets · TRL · PEFT · QLoRA · Kaggle GPU · FastAPI · pytest · GitHub Actions · Matplotlib

## Key Engineering Takeaways

- Built an end-to-end workflow connecting data preparation, model training, evaluation, inference architecture, and cost analysis.
- Learned how to fine-tune a quantized language model within limited GPU resources.
- Identified a practical limitation of generative intent classification: plausible predictions may not match the required label taxonomy.
- Designed fallback behavior to handle invalid model outputs and provider errors.
- Separated experimentally measured results from hypothetical deployment economics.

## Scope and Limitations

The completed training experiment is **gold-label supervised fine-tuning**, not teacher-supervised knowledge distillation.

The Claude teacher integration is implemented, but live teacher labeling was not completed because paid API credits were unavailable. Inference routing was demonstrated using mock providers, and no production serving or teacher-versus-student benchmark was performed.

The model's reported classification metrics come from the development validation set, not the official BANKING77 test set.

## Conclusion

The project demonstrates a practical foundation for building specialized LLM systems with limited computing resources. It combines model fine-tuning with software engineering, validation, routing, automated testing, and economic reasoning, while documenting the difference between implemented functionality and future extensions.