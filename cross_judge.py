"""Run a fixed pair (or set) of judges on a SHARED filtered conversation pool,
then report inter-judge agreement.

Design (per discussion 2026-05-07)
----------------------------------
Family overlap between judge and judged model is a confound: a Claude judge
grading a Claude conversation may be biased relative to grading a Llama
conversation. To mitigate:

1. Pick a small set of judges (default: TWO) from disjoint families.
   Default pair below uses one closed and one open-weights judge:
     - anthropic:claude-haiku-4-5    (closed, Anthropic family)
     - lambda:qwen3-32b-fp8          (open, Qwen/Alibaba family, via Lambda Labs)
   Both families are uncommon-or-absent in LMSYS-Chat-1M, so most LMSYS
   conversations remain in the pool after exclusion.

2. Build a SHARED conversation pool. Filter the source dataset's 'model' field
   (the LMSYS source-LLM identifier) to drop any conversation whose source
   model belongs to ANY judge's family.

3. Apply ALL judges to the SAME pool — so judge-vs-judge variance is a clean
   signal of inter-rater reliability (convergent validity), not driven by
   different judges seeing different conversations.

4. Report pairwise Cohen's quadratic-weighted kappa per marker, plus
   Pearson r between judges' aggregate scores.

Usage
-----
Default (anthropic + lambda Qwen, 100 LMSYS conversations):

  python cross_judge.py --source hf --data lmsys/lmsys-chat-1m --n 100 \\
      --out-dir results/cross_judge/

Custom judges:

  python cross_judge.py --source hf --data lmsys/lmsys-chat-1m --n 100 \\
      --judges anthropic:claude-haiku-4-5 openai:gpt-4o-mini \\
      --out-dir results/cross_judge_closed_only/

On synthetic examples (no source-model filtering needed):

  python cross_judge.py --source synthetic --data data/synthetic_examples.json \\
      --out-dir results/cross_judge_synthetic/
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from grader import MARKERS, GraderConfig, aggregate_score, grade_conversation
from run_experiment import load_hf_dataset, load_jsonl, load_synthetic


DEFAULT_JUDGES = [
    "anthropic:claude-haiku-4-5",  # closed, Anthropic family
    "lambda:qwen3-32b-fp8",        # open weights, Qwen/Alibaba family, via Lambda
]


def resolve_judges(specs: list[str]) -> list[GraderConfig]:
    """Parse 'provider[:model]' specs into GraderConfig list. Skips judges whose API key is missing."""
    out = []
    for spec in specs:
        if ":" in spec:
            provider, model = spec.split(":", 1)
        else:
            provider, model = spec, ""
        cfg = GraderConfig(provider=provider.strip(), model=model.strip())
        if not os.environ.get(cfg.api_key_env):
            print(f"Skipping judge {cfg.slug()}: {cfg.api_key_env} not set in environment.")
            continue
        out.append(cfg)
    return out


def grade_with_judge(convos: list[dict], cfg: GraderConfig, out_path: Path,
                     sleep: float, no_resume: bool) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    if out_path.exists() and not no_resume:
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen_ids.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                pass
        if seen_ids:
            print(f"  resuming {cfg.slug()}: {len(seen_ids)} already done")
    mode = "a" if (seen_ids and not no_resume) else "w"

    n_ok, n_fail = 0, 0
    with out_path.open(mode) as f:
        for i, c in enumerate(convos):
            cid = str(c["id"])
            if cid in seen_ids:
                continue
            try:
                result = grade_conversation(c["turns"], cfg)
                row = {
                    "id": cid,
                    "turns": c["turns"],
                    "n_turns": len(c["turns"]),
                    "scores": result["scores"],
                    "aggregate": aggregate_score(result["scores"]),
                    "raw_response": result["raw_response"],
                    "model": result["model"],
                    "provider": result["provider"],
                    "judge_slug": cfg.slug(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                if c.get("source_model"):
                    row["source_model"] = c["source_model"]
                if c.get("label_hint"):
                    row["label_hint"] = c["label_hint"]
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_ok += 1
                print(f"  [{cfg.slug()}] [{i+1}/{len(convos)}] id={cid} agg={row['aggregate']:.2f}")
            except Exception as e:
                n_fail += 1
                print(f"  [{cfg.slug()}] [{i+1}/{len(convos)}] FAIL id={cid} {type(e).__name__}: {e}")
            time.sleep(sleep)
    print(f"  {cfg.slug()}: OK={n_ok} FAIL={n_fail}")
    return n_ok


def _to_int(x):
    if x is None or x == "NA":
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def compute_pairwise_agreement(judge_files: dict[str, Path]) -> dict:
    """Pairwise quadratic-weighted Cohen's kappa per marker, plus aggregate-score Pearson r."""
    judge_dfs: dict[str, pd.DataFrame] = {}
    for slug, path in judge_files.items():
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "scores" not in r:
                continue
            row = {"id": str(r["id"]), "aggregate": r.get("aggregate")}
            for m in MARKERS:
                row[m] = _to_int(r["scores"].get(m, {}).get("score"))
            rows.append(row)
        judge_dfs[slug] = pd.DataFrame(rows)

    judges = list(judge_dfs.keys())
    summary = {
        "judges": judges,
        "n_per_judge": {j: len(judge_dfs[j]) for j in judges},
        "pairs": {},
    }

    for j1, j2 in combinations(judges, 2):
        merged = judge_dfs[j1].merge(judge_dfs[j2], on="id", suffixes=(f"_{j1}", f"_{j2}"))
        per_marker = {}
        for m in MARKERS:
            valid = merged[[f"{m}_{j1}", f"{m}_{j2}"]].dropna()
            n = len(valid)
            if n < 5:
                per_marker[m] = {"n": n, "kappa_quadratic": None, "exact_agreement": None,
                                 "note": "fewer than 5 non-NA pairs"}
                continue
            a = valid[f"{m}_{j1}"].astype(int)
            b = valid[f"{m}_{j2}"].astype(int)
            if a.nunique() < 2 or b.nunique() < 2:
                per_marker[m] = {"n": int(n), "kappa_quadratic": None,
                                 "exact_agreement": float((a == b).mean()),
                                 "note": "degenerate (no variance in one rater)"}
                continue
            per_marker[m] = {
                "n": int(n),
                "kappa_quadratic": float(cohen_kappa_score(a, b, weights="quadratic")),
                "exact_agreement": float((a == b).mean()),
            }
        agg_pair = merged[[f"aggregate_{j1}", f"aggregate_{j2}"]].dropna()
        if len(agg_pair) >= 5 and agg_pair.iloc[:, 0].nunique() >= 2 and agg_pair.iloc[:, 1].nunique() >= 2:
            corr = float(agg_pair.corr().iloc[0, 1])
        else:
            corr = None
        summary["pairs"][f"{j1} vs {j2}"] = {
            "n_overlap": int(len(merged)),
            "aggregate_pearson_r": corr,
            "per_marker": per_marker,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "synthetic", "jsonl"], default="hf")
    ap.add_argument("--data", default="lmsys/lmsys-chat-1m",
                    help="HF dataset name, or path to synthetic JSON / saved JSONL.")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES,
                    help=f"provider[:model] specs. Default: {DEFAULT_JUDGES}")
    ap.add_argument("--out-dir", default="results/cross_judge")
    ap.add_argument("--exclude-model-regex", default=None,
                    help="Override the auto-computed family-exclusion regex (HF source only).")
    ap.add_argument("--include-model-regex", default=None,
                    help="Restrict pool to source models matching this regex (HF source only).")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    # Resolve judges
    judges = resolve_judges(args.judges)
    if not judges:
        raise SystemExit("No usable judges (all API keys missing). Check .env.")
    print(f"Resolved {len(judges)} judges:")
    for j in judges:
        print(f"  - {j.slug():40s}  family-regex: {j.family_regex()}")

    # Build the SHARED exclude regex from the union of all judges' families
    if args.source == "hf":
        if args.exclude_model_regex:
            combined_exclude = args.exclude_model_regex
            print(f"Using user-supplied exclude regex: {combined_exclude}")
        else:
            family_regexes = [j.family_regex() for j in judges]
            combined_exclude = "|".join(f"({r})" for r in family_regexes)
            print(f"Auto-computed exclude regex (union of judge families):\n  {combined_exclude}")
    else:
        combined_exclude = None

    # Load conversations
    if args.source == "synthetic":
        convos = load_synthetic(Path(args.data))
    elif args.source == "jsonl":
        convos = load_jsonl(Path(args.data))
    else:
        convos = load_hf_dataset(
            args.data,
            args.hf_split,
            args.n,
            include_model_regex=args.include_model_regex,
            exclude_model_regex=combined_exclude,
        )
    convos = convos[: args.n]
    print(f"\nFinal conversation pool: {len(convos)} conversations.")

    # Cache the SHARED pool so all judges grade exactly the same set
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "conversation_pool.jsonl"
    with pool_path.open("w") as f:
        for c in convos:
            f.write(json.dumps(c) + "\n")
    print(f"Cached shared pool -> {pool_path}")

    # Save run metadata
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "data": args.data,
        "hf_split": args.hf_split,
        "n_requested": args.n,
        "n_pool": len(convos),
        "exclude_model_regex": combined_exclude,
        "include_model_regex": args.include_model_regex,
        "judges": [{"slug": j.slug(), "provider": j.provider, "model": j.model,
                    "family_regex": j.family_regex()} for j in judges],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    # Run each judge on the shared pool
    judge_files: dict[str, Path] = {}
    for j in judges:
        out_path = out_dir / f"{j.slug()}.jsonl"
        print(f"\n=== Judge: {j.slug()}  ->  {out_path} ===")
        grade_with_judge(convos, j, out_path, sleep=args.sleep, no_resume=args.no_resume)
        judge_files[j.slug()] = out_path

    # Pairwise agreement
    print(f"\n=== Pairwise agreement across {len(judges)} judges ===")
    summary = compute_pairwise_agreement(judge_files)
    summary_path = out_dir / "agreement.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {summary_path}")
    print(f"Pool:  {pool_path}")
    print(f"Meta:  {out_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
