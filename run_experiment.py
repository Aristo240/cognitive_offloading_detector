"""Run the cognitive offloading grader over a dataset of conversations.

Sources supported:
  --source synthetic         JSON file with a list of {id, turns} entries.
  --source hf                A HuggingFace dataset name.
  --source jsonl             A previously-saved JSONL (e.g. an output from this
                             script, or a cache file).

Each output row is fully self-contained: it includes the conversation turns,
parsed scores, raw model response, prompt, and metadata. This means you can
re-analyze, re-parse, or re-grade with a different rubric WITHOUT re-querying
the source dataset or the API.

Re-running the same command resumes from where it left off (skips already-graded
IDs in the output file). Use --no-resume to force a fresh run.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from grader import GraderConfig, aggregate_score, grade_conversation


def load_synthetic(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    out = []
    for r in rows:
        if "id" not in r or "turns" not in r:
            continue
        out.append({
            "id": str(r["id"]),
            "turns": r["turns"],
            "label_hint": r.get("label_hint"),
        })
    return out


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL produced by this script (or any file with id+turns rows)."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in r or "turns" not in r:
            continue
        out.append({
            "id": str(r["id"]),
            "turns": r["turns"],
            "label_hint": r.get("label_hint"),
        })
    return out


def load_hf_dataset(name: str, split: str, n: int, min_turns: int = 4) -> list[dict]:
    """Load conversations from a HuggingFace dataset.

    Tolerant to several common schemas (conversation, messages, turns).
    Filters short conversations.
    """
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=True)
    out = []
    seen_ids = set()
    for i, ex in enumerate(ds):
        if len(out) >= n:
            break
        turns = _extract_turns(ex)
        if turns is None or len(turns) < min_turns:
            continue
        cid = str(ex.get("conversation_id") or ex.get("id") or i)
        if cid in seen_ids:
            cid = f"{cid}_{i}"
        seen_ids.add(cid)
        out.append({"id": cid, "turns": turns})
    if not out:
        raise RuntimeError(
            f"Loaded 0 usable conversations from {name!r}. "
            "The dataset schema may not match. Inspect a few examples and adapt _extract_turns."
        )
    return out


def _extract_turns(ex: dict) -> list[dict] | None:
    """Pull a flat list of {role, content} turns from a HF example."""
    for key in ("conversation", "messages", "turns", "dialogue"):
        if key in ex and isinstance(ex[key], list):
            try:
                turns = []
                for t in ex[key]:
                    if not isinstance(t, dict):
                        continue
                    role = t.get("role") or t.get("from")
                    content = t.get("content") or t.get("value") or t.get("text")
                    if role and content:
                        turns.append({"role": str(role), "content": str(content)})
                if turns:
                    return turns
            except (TypeError, KeyError):
                continue
    return None


def cache_conversations(convos: list[dict], cache_path: Path) -> None:
    """Write loaded conversations to a JSONL cache (defense against re-querying)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        for c in convos:
            f.write(json.dumps(c) + "\n")
    print(f"Cached {len(convos)} loaded conversations -> {cache_path}")


def read_existing_ids(out_path: Path) -> set[str]:
    """For resumability: which IDs are already in the output file?"""
    if not out_path.exists():
        return set()
    seen = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if "id" in r:
                seen.add(str(r["id"]))
        except json.JSONDecodeError:
            continue
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "hf", "jsonl"], default="synthetic")
    ap.add_argument("--data", default="data/synthetic_examples.json",
                    help="Path to JSON/JSONL or HF dataset name.")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="results/grades.jsonl")
    ap.add_argument("--cache", default=None,
                    help="Path to write loaded conversations cache. Default: derived from --out.")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--sleep", type=float, default=0.3, help="Seconds between API calls.")
    ap.add_argument("--no-resume", action="store_true",
                    help="Overwrite output file. Default is to append/resume.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out_path.parent / f"{out_path.stem}_conversations.jsonl"

    # Load conversations
    if args.source == "synthetic":
        convos = load_synthetic(Path(args.data))
    elif args.source == "jsonl":
        convos = load_jsonl(Path(args.data))
    else:
        convos = load_hf_dataset(args.data, args.hf_split, args.n)
    convos = convos[: args.n]
    print(f"Loaded {len(convos)} conversations from source={args.source} ({args.data})")

    # Cache the loaded conversations so we never need to re-query
    cache_conversations(convos, cache_path)

    # Resumability
    seen_ids: set[str] = set()
    if not args.no_resume:
        seen_ids = read_existing_ids(out_path)
        if seen_ids:
            print(f"Resuming: {len(seen_ids)} conversations already graded; will skip them.")
    mode = "a" if (seen_ids and not args.no_resume) else "w"

    cfg = GraderConfig(provider=args.provider, model=args.model)

    n_ok, n_fail, n_skip = 0, 0, 0
    with out_path.open(mode) as f:
        for i, c in enumerate(convos):
            cid = str(c["id"])
            if cid in seen_ids:
                n_skip += 1
                print(f"[{i+1:>3}/{len(convos)}] SKIP id={cid} (already graded)")
                continue
            try:
                result = grade_conversation(c["turns"], cfg)
                agg = aggregate_score(result["scores"])
                row = {
                    "id": cid,
                    "turns": c["turns"],
                    "n_turns": len(c["turns"]),
                    "scores": result["scores"],
                    "aggregate": agg,
                    "raw_response": result["raw_response"],
                    "prompt": result["prompt"],
                    "model": result["model"],
                    "provider": result["provider"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                if c.get("label_hint"):
                    row["label_hint"] = c["label_hint"]
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_ok += 1
                print(f"[{i+1:>3}/{len(convos)}] id={cid:<25} agg={agg:.2f}")
            except KeyboardInterrupt:
                print("\nInterrupted. Partial results saved; rerun the same command to resume.")
                raise
            except Exception as e:
                n_fail += 1
                print(f"[{i+1:>3}/{len(convos)}] FAIL id={cid:<25} {type(e).__name__}: {e}")
            time.sleep(args.sleep)

    print(f"\nDone. OK={n_ok} SKIP={n_skip} FAIL={n_fail}")
    print(f"  Output:        {out_path}")
    print(f"  Conv. cache:   {cache_path}")


if __name__ == "__main__":
    main()
