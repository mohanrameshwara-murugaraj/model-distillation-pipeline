
"""Local Qwen3 + LoRA student inference.

Requires transformers, torch and peft.
Loads the trained adapter without modifying the base model.
"""

from time import perf_counter

from .providers import Prediction


# Exact system prompt used during Kaggle training.
TRAINING_SYSTEM = (
    "You are a banking customer support intent classifier. "
    "Choose the single best banking intent label. "
    "Return only the intent label with no explanation."
)


class LocalQwenStudent:
    def __init__(
        self,
        adapter_path="models/qwen3-banking77-light-final",
        base_model="Qwen/Qwen3-1.7B",
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = f"{base_model}+lora"

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(
            base, adapter_path
        )
        self.model.eval()

    def predict(self, text: str) -> Prediction:
        started = perf_counter()

        messages = [
            {"role": "system", "content": TRAINING_SYSTEM},
            {"role": "user", "content": text},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(self.model.device)

        with self.torch.inference_mode():
            output = self.model.generate(
                inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        tokens = output[0, inputs.shape[1]:]
        intent = self.tokenizer.decode(
            tokens, skip_special_tokens=True
        ).strip()

        return Prediction(
            intent=intent,
            latency_ms=(perf_counter() - started) * 1000,
            input_tokens=int(inputs.shape[1]),
            output_tokens=int(tokens.shape[0]),
            model=self.name,
        )
