"""Evaluate saved LoRA checkpoints on validation, log accuracy/examples to W&B.

This is intentionally a separate step from training loss evaluation; it uses
actual generated labels rather than proxy next-token loss. Requires CUDA.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--validation", default="data/processed/validation.jsonl")
    parser.add_argument("--labels", default="data/processed/labels.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint evaluation requires an NVIDIA CUDA GPU")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from distill_lab.taxonomy import STUDENT_SYSTEM, load_labels

    val = read_jsonl(args.validation)[:args.limit]
    if not val:
        raise ValueError("No validation examples")
    labels = set(load_labels(args.labels))
    ckpts = sorted(Path(args.run_dir).glob("checkpoint-*"),
                   key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    if not ckpts:
        raise ValueError("No checkpoint-* folders in run directory")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    session = None
    if args.wandb:
        import wandb
        session = wandb.init(project="banking77-distillation-validation",
                             name=Path(args.run_dir).name)
    reports = []
    for ckpt in ckpts:
        # Reload base per checkpoint so adapters cannot mutate a shared model.
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported()
                                       else torch.float16),
            ),
        )
        model = PeftModel.from_pretrained(base, str(ckpt))
        model.eval()
        output_rows = []
        for row in val:
            messages = [{"role": "system", "content": STUDENT_SYSTEM},
                        {"role": "user", "content": row["text"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**tokens, max_new_tokens=64, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            predicted = tokenizer.decode(output[0, tokens["input_ids"].shape[1]:],
                                         skip_special_tokens=True).strip()
            output_rows.append({
                "text": row["text"], "gold": row["gold_intent"],
                "predicted": predicted if predicted in labels else None,
                "valid": predicted in labels,
            })
        accuracy = sum(r["gold"] == r["predicted"] for r in output_rows) / len(output_rows)
        record = {"checkpoint": ckpt.name, "validation_examples": len(output_rows),
                  "accuracy": round(accuracy, 6),
                  "invalid": sum(not r["valid"] for r in output_rows),
                  "samples": output_rows[:5]}
        reports.append(record)
        if session:
            import wandb
            step = int(ckpt.name.rsplit("-", 1)[-1])
            wandb.log({"validation/generated_accuracy": accuracy,
                       "validation/sample_outputs": wandb.Table(
                           columns=["query", "gold", "predicted"],
                           data=[[r["text"], r["gold"], r["predicted"]]
                                 for r in output_rows[:5]])}, step=step)
        del model, base
        torch.cuda.empty_cache()
    report_path = Path(args.run_dir) / "checkpoint_validation.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in report.items() if k != "samples"}
                      for report in reports], indent=2))
    if session:
        session.finish()


if __name__ == "__main__":
    main()
