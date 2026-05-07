"""Distribution plots and summary stats for cognitive offloading grades.

Reads JSONL from run_experiment.py and writes:
  - results/grades.csv          flat table
  - results/distributions.png   histogram + per-marker means
  - prints summary to stdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in headless / venv contexts
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MARKERS = [
    "answer_copying",
    "no_elaboration",
    "no_error_correction",
    "no_questioning",
    "verbatim_reuse",
]


def _to_int(x):
    if x == "NA" or x is None:
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="input", required=True, help="JSONL output from run_experiment.py")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    rows = []
    for line in Path(args.input).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise SystemExit(f"No valid rows found in {args.input}")

    valid_rows = [r for r in rows if "scores" in r and all(m in r["scores"] for m in MARKERS)]
    skipped = len(rows) - len(valid_rows)
    if skipped:
        print(f"Note: skipped {skipped} rows missing scores or markers.")
    if not valid_rows:
        raise SystemExit("No rows with complete scores. Cannot analyze.")

    df = pd.DataFrame([
        {
            "id": r["id"],
            "aggregate": r.get("aggregate"),
            "n_turns": r.get("n_turns", len(r.get("turns", []))),
            **{m: _to_int(r["scores"][m]["score"]) for m in MARKERS},
        }
        for r in valid_rows
    ])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "grades.csv", index=False)

    print(f"=== Sample: {len(df)} conversations ===\n")

    print("=== Per-marker statistics ===")
    print(f"{'marker':<22} {'mean':>6} {'strong-rate':>12} {'NA-rate':>9} {'n':>5}")
    for m in MARKERS:
        vals = df[m].dropna()
        n_na = df[m].isna().sum()
        if len(vals) == 0:
            print(f"  {m:<20} {'-':>6} {'-':>12} {n_na/len(df):>9.0%} {len(vals):>5}")
            continue
        prop_strong = (vals >= 2).mean()
        print(f"  {m:<20} {vals.mean():>6.2f} {prop_strong:>11.0%} {n_na/len(df):>9.0%} {len(vals):>5}")

    print(f"\n=== Aggregate offloading score ===")
    agg = df["aggregate"].dropna()
    if len(agg):
        print(f"  mean={agg.mean():.2f}  median={agg.median():.2f}  sd={agg.std():.2f}  n={len(agg)}")
        print(f"  thinking-with-AI   (<0.5):     {(agg < 0.5).mean():.0%}")
        print(f"  mixed              (0.5-1.0):  {((agg >= 0.5) & (agg < 1.0)).mean():.0%}")
        print(f"  offloading-dominant (>=1.0):   {(agg >= 1.0).mean():.0%}")

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    if len(agg):
        ax[0].hist(agg, bins=20, edgecolor="black", color="#4C72B0")
        ax[0].axvline(0.5, ls="--", color="grey", alpha=0.5)
        ax[0].axvline(1.0, ls="--", color="grey", alpha=0.5)
        ax[0].set_title(f"Aggregate offloading score (n={len(agg)})")
        ax[0].set_xlabel("score (0 = thinking-with, 2 = offloading)")
        ax[0].set_ylabel("count")

    marker_means = [df[m].dropna().mean() if df[m].notna().any() else 0 for m in MARKERS]
    ax[1].bar(range(len(MARKERS)), marker_means, color="#55A868")
    ax[1].set_xticks(range(len(MARKERS)))
    ax[1].set_xticklabels([m.replace("_", "\n") for m in MARKERS], fontsize=8)
    ax[1].set_title("Mean score per marker")
    ax[1].set_ylim(0, 2)
    ax[1].axhline(1.0, ls="--", color="grey", alpha=0.5)

    fig.tight_layout()
    fig.savefig(outdir / "distributions.png", dpi=150)
    print(f"\nSaved: {outdir/'grades.csv'}")
    print(f"Saved: {outdir/'distributions.png'}")


if __name__ == "__main__":
    main()
