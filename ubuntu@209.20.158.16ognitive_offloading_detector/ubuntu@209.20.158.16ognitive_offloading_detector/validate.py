"""Compute inter-rater agreement between the LLM grader and human hand-labels.

Inputs:
  --llm    Path to grader output JSONL from run_experiment.py.
  --human  Path to a CSV with columns: id, answer_copying, no_elaboration,
           no_error_correction, no_questioning, verbatim_reuse.
           Cells may be 0/1/2 or "NA".

Output:
  results/validation.json  with Cohen's quadratic-weighted kappa per marker
                           and exact-agreement rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

MARKERS = [
    "answer_copying",
    "no_elaboration",
    "no_error_correction",
    "no_questioning",
    "verbatim_reuse",
]


def _to_int(x):
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    if isinstance(x, str):
        s = x.strip().upper()
        if s in ("NA", "N/A", ""):
            return None
        try:
            return int(s)
        except ValueError:
            return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", required=True, help="JSONL output from run_experiment.py")
    ap.add_argument("--human", required=True, help="Human-labeled CSV")
    ap.add_argument("--out", default="results/validation.json")
    args = ap.parse_args()

    llm_rows = []
    for line in Path(args.llm).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            llm_rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not llm_rows:
        raise SystemExit(f"No valid rows found in {args.llm}")

    llm_df = pd.DataFrame([
        {"id": str(r["id"]), **{m: _to_int(r["scores"][m]["score"]) for m in MARKERS}}
        for r in llm_rows
        if "scores" in r and all(m in r["scores"] for m in MARKERS)
    ])

    human_df = pd.read_csv(args.human, dtype=str)
    human_df.columns = [c.strip() for c in human_df.columns]  # tolerate trailing spaces
    missing = [c for c in ["id", *MARKERS] if c not in human_df.columns]
    if missing:
        raise SystemExit(
            f"Human-labels CSV is missing columns: {missing}. "
            f"Found: {list(human_df.columns)}"
        )
    human_df["id"] = human_df["id"].astype(str).str.strip()
    for m in MARKERS:
        human_df[m] = human_df[m].apply(_to_int)

    merged = llm_df.merge(human_df, on="id", suffixes=("_llm", "_human"))
    print(f"Matched conversations: {len(merged)}")

    results: dict[str, dict] = {"n_matched": int(len(merged)), "markers": {}}

    for m in MARKERS:
        valid = merged[[f"{m}_llm", f"{m}_human"]].dropna()
        n = len(valid)
        if n < 5:
            results["markers"][m] = {"n": n, "kappa_quadratic": None, "exact_agreement": None,
                                     "note": "fewer than 5 non-NA pairs; skipping"}
            continue
        a = valid[f"{m}_llm"].astype(int)
        b = valid[f"{m}_human"].astype(int)
        # Need non-degenerate variance for kappa to be defined
        if a.nunique() < 2 or b.nunique() < 2:
            kappa = None
            note = "degenerate (all same value in one rater); kappa undefined"
        else:
            kappa = float(cohen_kappa_score(a, b, weights="quadratic"))
            note = None
        results["markers"][m] = {
            "n": int(n),
            "kappa_quadratic": kappa,
            "exact_agreement": float((a == b).mean()),
            "note": note,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
