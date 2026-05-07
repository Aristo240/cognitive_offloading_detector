"""Re-grade conversations from a saved JSONL without re-querying the source.

Use cases:
  - You changed rubric.md and want to re-score with the new rubric.
  - You want to compare two grader models on the same conversations.
  - The original parsing failed and you want to re-parse the saved raw_response
    (use --reparse-only to do this without spending API tokens).

Each row in the input must contain an 'id' and 'turns' field, which is the
default for run_experiment.py output.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from grader import (
    GraderConfig,
    _extract_json,
    _validate_scores,
    aggregate_score,
    grade_conversation,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="input", required=True, help="JSONL input (with id+turns).")
    ap.add_argument("--out", required=True, help="JSONL output.")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--reparse-only", action="store_true",
                    help="Do not query the API; just re-parse saved raw_response fields.")
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
    print(f"Loaded {len(rows)} rows from {args.input}")

    cfg = GraderConfig(provider=args.provider, model=args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail = 0, 0
    with out_path.open("w") as f:
        for i, r in enumerate(rows):
            cid = r.get("id", f"row_{i}")
            try:
                if args.reparse_only:
                    raw = r.get("raw_response")
                    if not raw:
                        raise ValueError("Row has no raw_response to re-parse.")
                    scores = _extract_json(raw)
                    _validate_scores(scores)
                    new = {**r, "scores": scores, "aggregate": aggregate_score(scores),
                           "regraded_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                           "regrade_method": "reparse_only"}
                else:
                    if "turns" not in r:
                        raise ValueError("Row has no turns; cannot re-grade.")
                    result = grade_conversation(r["turns"], cfg)
                    new = {
                        **r,
                        "scores": result["scores"],
                        "aggregate": aggregate_score(result["scores"]),
                        "raw_response": result["raw_response"],
                        "prompt": result["prompt"],
                        "model": result["model"],
                        "provider": result["provider"],
                        "regraded_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "regrade_method": f"api:{args.provider}/{args.model}",
                    }
                f.write(json.dumps(new) + "\n")
                f.flush()
                n_ok += 1
                print(f"[{i+1:>3}/{len(rows)}] id={cid:<25} agg={new['aggregate']:.2f}")
            except Exception as e:
                n_fail += 1
                print(f"[{i+1:>3}/{len(rows)}] FAIL id={cid:<25} {type(e).__name__}: {e}")
            if not args.reparse_only:
                time.sleep(args.sleep)

    print(f"\nDone. OK={n_ok} FAIL={n_fail}. Output: {out_path}")


if __name__ == "__main__":
    main()
