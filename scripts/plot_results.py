
from pathlib import Path
import json
import matplotlib.pyplot as plt

out = Path("docs/figures")
out.mkdir(parents=True, exist_ok=True)

with open("results/experiment_summary.json") as f:
    experiment = json.load(f)

with open("results/economics_scenario.json") as f:
    economics = json.load(f)["results"]

# 1. Training loss
steps = [100, 200]
training_loss = [0.478200, 0.198700]
validation_loss = [0.441807, 0.216142]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(steps, training_loss, marker="o", label="Training")
ax.plot(steps, validation_loss, marker="o", label="Validation")
ax.set(title="Qwen3-1.7B QLoRA — Loss",
       xlabel="Optimizer step", ylabel="Loss")
ax.set_xticks(steps)
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out / "training_loss.png", dpi=180)
plt.close(fig)

# 2. Validation prediction outcomes
total = experiment["validation_examples_evaluated"]
correct = round(total * experiment["validation_accuracy"])
invalid = experiment["invalid_predictions"]
valid_wrong = total - correct - invalid

fig, ax = plt.subplots(figsize=(7, 4.5))
names = ["Correct", "Valid but wrong", "Invalid label"]
values = [correct, valid_wrong, invalid]
bars = ax.bar(names, values)
ax.bar_label(bars)
ax.set(title="Validation prediction outcomes",
       ylabel="Number of predictions")
ax.set_ylim(0, max(values) * 1.2)
fig.tight_layout()
fig.savefig(out / "validation_predictions.png", dpi=180)
plt.close(fig)

# 3. Economics — hypothetical scenario
costs = economics["at_requested_volume"]

fig, ax = plt.subplots(figsize=(7, 4.5))
names = ["Teacher-only", "Student + fallback"]
values = [
    costs["teacher_daily_usd"],
    costs["student_router_daily_usd"]
]
bars = ax.bar(names, values)
ax.bar_label(bars, fmt="$%.2f")
ax.set(
    title="Hypothetical cost at 10,000 requests/day",
    ylabel="USD per day"
)
ax.set_ylim(0, max(values) * 1.2)
fig.tight_layout()
fig.savefig(out / "economics_scenario.png", dpi=180)
plt.close(fig)

print("Charts saved to:", out)
