import json
import matplotlib.pyplot as plt
import numpy as np

with open("benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

models = {}
for name, results in data["models"].items():
    valid = [r for r in results if r["status"] == "ok"]
    if not valid:
        continue
    models[name] = {
        "total": np.mean([r["elapsed"] for r in valid]),
        "ttft": np.mean([r["ttft"] for r in valid]),
        "gen": np.mean([r["gen"] for r in valid]),
    }

# Sort by total time
names = sorted(models, key=lambda n: models[n]["total"])
total = [models[n]["total"] for n in names]
ttft = [models[n]["ttft"] for n in names]
gen = [models[n]["gen"] for n in names]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = [
    ("Total Time (Question → Complete)", total),
    ("TTFT (Question → First Token)", ttft),
    ("Generation Time (First Token → Complete)", gen),
]

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

for ax, (title, vals) in zip(axes, metrics):
    bars = ax.barh(names, vals, color=colors[: len(names)])
    ax.set_xlabel("Seconds")
    ax.set_title(title, fontsize=12, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}s", va="center", fontsize=9)
    ax.margins(x=0.15)

plt.suptitle("Ollama Model Benchmark – Average across 29 questions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("benchmark_chart.png", dpi=150, bbox_inches="tight")
plt.savefig("benchmark_chart.svg", bbox_inches="tight")
print("Saved benchmark_chart.png / .svg")
