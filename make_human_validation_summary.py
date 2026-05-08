"""Build summary of human-vs-LLM-judge validation across all 3 judges."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/cross_judge_ultrachat"

JUDGES = [
    ("Anthropic Haiku", "validation_anthropic__claude-haiku-4-5.json", "#4C72B0"),
    ("Gemini 2.0 Flash", "validation_gemini__gemini-2.0-flash.json", "#55A868"),
    ("Llama-3.3-70B-FP8", "validation_vllm__Llama-3.3-70B-Instruct-FP8.json", "#C44E52"),
]
MARKERS_DEFINED = ["answer_copying", "no_elaboration", "no_questioning"]
ALL_MARKERS = ["answer_copying", "no_elaboration", "no_error_correction",
               "no_questioning", "verbatim_reuse"]

# Aggregate
rows_csv = []
plot_data = {m: [] for m in MARKERS_DEFINED}
for judge_name, fname, color in JUDGES:
    data = json.loads((RESULTS / fname).read_text())
    n = data["n_matched"]
    for m in ALL_MARKERS:
        info = data["markers"][m]
        kappa = info.get("kappa_quadratic")
        agree = info.get("exact_agreement")
        n_marker = info.get("n", 0)
        note = info.get("note") or ""
        rows_csv.append({
            "judge": judge_name,
            "marker": m,
            "n_pairs_with_both_scored": n_marker,
            "kappa_quadratic": f"{kappa:.4f}" if kappa is not None else "n/a",
            "exact_agreement": f"{agree:.4f}" if agree is not None else "n/a",
            "note": note,
        })
        if m in MARKERS_DEFINED:
            plot_data[m].append((judge_name, kappa, color))

# CSV
out_csv = RESULTS / "human_validation_summary.csv"
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["judge", "marker", "n_pairs_with_both_scored",
                                       "kappa_quadratic", "exact_agreement", "note"])
    w.writeheader()
    for r in rows_csv:
        w.writerow(r)
print(f"Saved: {out_csv}")

# Plot
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(11, 5.5))
markers = MARKERS_DEFINED
n_judges = len(JUDGES)
n_markers = len(markers)
x = np.arange(n_markers)
width = 0.26
for j_idx, (judge_name, _, color) in enumerate(JUDGES):
    vals = []
    for m in markers:
        for jn, k, c in plot_data[m]:
            if jn == judge_name:
                vals.append(k if k is not None else 0)
                break
        else:
            vals.append(0)
    is_null = []
    for m in markers:
        for jn, k, c in plot_data[m]:
            if jn == judge_name:
                is_null.append(k is None)
                break
        else:
            is_null.append(True)
    bars = ax.bar(x + (j_idx - 1) * width, vals, width, color=color, edgecolor="black",
                   label=judge_name)
    for i, (v, null) in enumerate(zip(vals, is_null)):
        label = "n.d." if null else f"{v:.2f}"
        ax.text(x[i] + (j_idx - 1) * width, v + 0.02 if v >= 0 else v - 0.05,
                label, ha="center", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(markers, fontsize=12)
ax.set_ylabel("Cohen's κ (quadratic-weighted) vs hand-coded human labels (n=30)", fontsize=12)
ax.set_title("Per-marker agreement with human labels — Anthropic Haiku tracks best\n"
             "(NEC and VR omitted: human marked NA on most rows so n_pairs<5 for κ)",
             fontsize=12)
ax.axhline(0, color="black", linewidth=0.6)
ax.axhline(0.4, linestyle="--", color="grey", alpha=0.6, label="moderate agreement (κ=0.4)")
ax.axhline(0.6, linestyle="--", color="grey", alpha=0.4, label="substantial (κ=0.6)")
ax.set_ylim(-0.05, 0.7)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
plot_path = RESULTS / "human_validation_kappa.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved: {plot_path}")

# Print headline
print("\n=== HUMAN-vs-JUDGE κ SUMMARY ===")
for m in MARKERS_DEFINED:
    print(f"\n{m}:")
    for jn, k, c in plot_data[m]:
        if k is None:
            print(f"  {jn:25s}: n.d. (degenerate or insufficient data)")
        else:
            print(f"  {jn:25s}: κ = {k:.3f}")
