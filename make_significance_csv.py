"""Compile cross-judge agreement statistics across all datasets into one CSV.

Note: Pearson r and Cohen's kappa here are descriptive measures of agreement,
not p-value tests. We include them as the primary inter-rater reliability
metrics. Where applicable we also test the convergent-validity claim with a
Fisher z-transformed CI on the Pearson r.
"""
from __future__ import annotations

import csv
import json
from math import atanh, sqrt, tanh, erf
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "significance_tests.csv"

DATASETS = {
    "synthetic": "cross_judge_synthetic/agreement.json",
    "WildChat-1M (n=100)": "cross_judge_wildchat/agreement.json",
    "UltraChat-200k (n=92-100)": "cross_judge_ultrachat/agreement.json",
}


def fisher_z_p(r: float, n: int) -> tuple[float, float, float]:
    """Test H0: rho = 0 via Fisher z. Returns (z, p_two_sided, ci_low, ci_high) approx."""
    if n < 4 or abs(r) >= 1:
        return float("nan"), float("nan"), float("nan"), float("nan")
    z = atanh(r)
    se = 1 / sqrt(n - 3)
    z_stat = z / se
    p = 2 * (1 - 0.5 * (1 + erf(abs(z_stat) / sqrt(2))))
    ci_low = tanh(z - 1.96 * se)
    ci_high = tanh(z + 1.96 * se)
    return z_stat, p, ci_low, ci_high


def main() -> None:
    rows = []
    for ds_name, rel in DATASETS.items():
        path = RESULTS / rel
        if not path.exists():
            continue
        ag = json.loads(path.read_text())
        for pair_key, info in ag["pairs"].items():
            r = info.get("aggregate_pearson_r")
            n = info.get("n_overlap")
            if r is None or n is None:
                continue
            z_stat, p, lo, hi = fisher_z_p(r, n)
            short = pair_key.replace("__", " ").replace("anthropic claude-haiku-4-5", "Anthropic Haiku") \
                            .replace("openai gpt-4o-mini", "OpenAI gpt-4o-mini") \
                            .replace("gemini gemini-2.0-flash", "Gemini 2.0 Flash") \
                            .replace("vllm Llama-3.3-70B-Instruct-FP8", "Llama-3.3-70B-FP8")
            ci_str = f"[{lo:.3f}, {hi:.3f}]" if lo == lo else "n/a"
            rows.append({
                "dataset": ds_name,
                "test_family": "inter-judge aggregate Pearson r (Fisher-z H0: r=0)",
                "test": short,
                "estimate_r": f"{r:+.4f}",
                "ci_95_pct": ci_str,
                "p_value_vs_zero": f"{p:.2e}" if p == p else "n/a",
                "n": n,
                "significant_at_0.05": "yes" if (p == p and p < 0.05) else "no",
                "interpretation": (
                    "convergent validity. Note: large n + r=0.17 still passes H0:r=0; "
                    "it does NOT mean judges agree well, only that they're not anticorrelated"
                ),
            })
            for marker, mk in info.get("per_marker", {}).items():
                kq = mk.get("kappa_quadratic")
                if kq is None:
                    continue
                rows.append({
                    "dataset": ds_name,
                    "test_family": "Cohen's quadratic kappa (descriptive)",
                    "test": f"{short}  |  marker={marker}",
                    "estimate_r": f"{kq:+.4f}",
                    "ci_95_pct": "n/a",
                    "p_value_vs_zero": "n/a",
                    "n": mk["n"],
                    "significant_at_0.05": "n/a",
                    "interpretation": "kappa <0.4 weak, 0.4-0.6 moderate, 0.6-0.8 substantial",
                })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "test_family", "test", "estimate_r",
                                          "ci_95_pct", "p_value_vs_zero", "n",
                                          "significant_at_0.05", "interpretation"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print("\nHeadline rows:")
    for ds_name in DATASETS:
        sub = [r for r in rows if r["dataset"] == ds_name and r["test_family"].startswith("inter-judge")]
        print(f"\n{ds_name}:")
        for row in sub:
            print(f"  {row['test'][:55]:55s}  r={row['estimate_r']:>8s}  CI={row['ci_95_pct']}  p={row['p_value_vs_zero']}")


if __name__ == "__main__":
    main()
