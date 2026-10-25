"""QLoRA/LoRA SFT on accepted Claude labels. Requires an actual NVIDIA GPU.

Run separate rank-8 and rank-32 experiments. Train never reads official test data.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/sft_train.jsonl")
    parser.add_argument("--validation", default="data/processed/sft_validation.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--out", default="outputs/train")
    parser.add_argument("--rank", type=int, required=True, choices=[8, 32])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--full-precision-lora", action="store_true",
                        help="Do not 4-bit quantize base model (higher VRAM)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required for Qwen3-8B training")
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    train = load_dataset("json", data_files=args.train, split="train")
    val = load_dataset("json", data_files=args.validation, split="train")
    if len(train) == 0:
        raise ValueError("Empty filtered teacher training file")
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = SFTConfig(
        output_dir=str(Path(args.out) / args.run_name),
        max_length=256,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        save_total_limit=4,
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.run_name,
        model_init_kwargs={"dtype": precision},
        gradient_checkpointing=True,
        bf16=(precision == torch.bfloat16),
        fp16=(precision == torch.float16),
        completion_only_loss=True,
        seed=42,
    )
    quant = None if args.full_precision_lora else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=precision,
    )
    adapter = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train,
        eval_dataset=val,
        peft_config=adapter,
        quantization_config=quant,
    )
    trainer.train()
    trainer.save_model(str(Path(args.out) / args.run_name / "final-adapter"))
    trainer.save_state()
    print(f"Adapter saved for {args.run_name}; verify quality separately on validation.")


if __name__ == "__main__":
    main()
