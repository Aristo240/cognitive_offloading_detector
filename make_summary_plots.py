"""Create summary plots for the README:

  1. 3-judge agreement bar chart on UltraChat (Anthropic vs Gemini vs Llama).
  2. Side-by-side histograms of aggregate offloading scores per judge on UltraChat.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "cross_judge_ultrachat"

# 1. Agreement bar chart
agreement = json.loads((RESULTS / "agreement.json").read_text())
pairs = agreement["pairs"]
labels = []
rs = []
ks_ac = []  # answer_copying kappa
for pair_label, info in pairs.items():
    short = pair_label.replace("anthropic__claude-haiku-4-5", "Anthropic Haiku") \
                       .replace("gemini__gemini-2.0-flash", "Gemini 2.0 Flash") \
                       .replace("vllm__Llama-3.3-70B-Instruct-FP8", "Llama-3.3-70B-FP8")
    labels.append(short)
    rs.append(info.get("aggregate_pearson_r") or 0)
    k = info["per_marker"].get("answer_copying", {}).get("kappa_quadratic")
    ks_ac.append(k if k is not None else 0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

x = np.arange(len(labels))
ax = axes[0]
bars = ax.bar(x, rs, color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
ax.set_ylabel("Aggregate Pearson r")
ax.set_title("3-judge cross-family agreement\n(UltraChat-200k, n=92-100)")
ax.set_ylim(0, 0.6)
ax.axhline(0.3, linestyle="--", color="grey", alpha=0.4, label="weak/moderate")
for i, v in enumerate(rs):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

ax = axes[1]
bars = ax.bar(x, ks_ac, color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
ax.set_ylabel("Cohen's κ (quadratic)")
ax.set_title("answer_copying agreement (κ)")
ax.set_ylim(-0.05, 0.6)
ax.axhline(0.4, linestyle="--", color="grey", alpha=0.4, label="moderate")
ax.axhline(0.0, color="black", linewidth=0.5)
for i, v in enumerate(ks_ac):
    ax.text(i, v + 0.01 if v >= 0 else v - 0.03, f"{v:.2f}",
            ha="center", fontsize=10, fontweight="bold")

fig.suptitle("Anthropic and Llama (different families) converge; Gemini diverges from both",
             fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "agreement_bars.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS / 'agreement_bars.png'}")


# 2. Per-judge histograms of aggregate scores on UltraChat
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
files = [
    ("Anthropic Haiku", "anthropic__claude-haiku-4-5.jsonl", "#4C72B0"),
    ("Gemini 2.0 Flash", "gemini__gemini-2.0-flash.jsonl", "#55A868"),
    ("Llama-3.3-70B-FP8 (open, vLLM)", "vllm__Llama-3.3-70B-Instruct-FP8.jsonl", "#C44E52"),
]
for ax, (name, fname, color) in zip(axes, files):
    path = RESULTS / fname
    if not path.exists():
        ax.set_title(f"{name}\n(file missing)")
        continue
    aggs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            v = r.get("aggregate")
            if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                aggs.append(float(v))
        except json.JSONDecodeError:
            continue
    aggs = np.array(aggs)
    ax.hist(aggs, bins=20, range=(0, 2), color=color, edgecolor="black", alpha=0.8)
    ax.set_xlim(0, 2)
    ax.set_xlabel("aggregate offloading score (0=think-with, 2=offload)")
    mean = aggs.mean() if len(aggs) else 0
    ax.set_title(f"{name}\nn={len(aggs)}, mean={mean:.2f}")
    ax.axvline(mean, linestyle="--", color="black", alpha=0.6, linewidth=1)
axes[0].set_ylabel("conversations")
fig.suptitle("Per-judge distributions of offloading scores on the SAME 100 UltraChat conversations",
             fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS / "per_judge_distributions.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS / 'per_judge_distributions.png'}")
