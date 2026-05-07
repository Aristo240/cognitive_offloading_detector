"""Build a hand-coding worksheet: 30 UltraChat conversations stratified by
Anthropic LLM-aggregate score, plus a blank label CSV with the same IDs.

Outputs (in validation/):
  - conversations_to_code.md    : readable conversations, no LLM scores shown
  - human_labels_to_fill.csv    : blank score template keyed by conversation id
  - sampling_metadata.json      : record of which IDs were sampled and why

Run from the project root:
  python3 validation/build_coding_sheet.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "results/cross_judge_ultrachat/conversation_pool.jsonl"
ANTHRO = ROOT / "results/cross_judge_ultrachat/anthropic__claude-haiku-4-5.jsonl"
OUT_MD = ROOT / "validation/conversations_to_code.md"
OUT_CSV = ROOT / "validation/human_labels_to_fill.csv"
OUT_META = ROOT / "validation/sampling_metadata.json"

N_PER_BIN = 10  # 10 low + 10 mid + 10 high = 30
SEED = 7

MARKERS = ["answer_copying", "no_elaboration", "no_error_correction",
           "no_questioning", "verbatim_reuse"]


def load_jsonl(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            out[str(r["id"])] = r
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def main():
    random.seed(SEED)
    pool = load_jsonl(POOL)
    anthro = load_jsonl(ANTHRO)
    print(f"Pool: {len(pool)} conversations.  Anthropic graded: {len(anthro)}.")

    # Stratify by Anthropic aggregate
    rows = []
    for cid, r in anthro.items():
        agg = r.get("aggregate")
        if agg is None or cid not in pool:
            continue
        rows.append((cid, agg))

    rows.sort(key=lambda x: x[1])
    n = len(rows)
    low_cut = n // 3
    high_cut = 2 * n // 3
    low = rows[:low_cut]
    mid = rows[low_cut:high_cut]
    high = rows[high_cut:]
    print(f"Stratification: low={len(low)} mid={len(mid)} high={len(high)}")

    sampled = []
    for label, bin_ in [("low", low), ("mid", mid), ("high", high)]:
        chosen = random.sample(bin_, min(N_PER_BIN, len(bin_)))
        sampled.extend([(cid, agg, label) for cid, agg in chosen])

    sampled.sort(key=lambda x: (x[2], x[1]))  # group by bin, then by score
    print(f"Sampled: {len(sampled)} conversations")

    # Shuffle order so adjacent items aren't from the same bin (further
    # reduces any cuing from sample order).
    shuffled = list(sampled)
    random.Random(SEED + 1).shuffle(shuffled)

    # Markdown file: blind to LLM scores AND bins.
    md = ["# Hand-coding worksheet — UltraChat n=30",
          "",
          f"Sampled {len(shuffled)} conversations from the UltraChat-200k cross-judge pool. Order randomized; bin/score information is intentionally not shown to keep your labels blind to the LLM grader's scores.",
          "",
          "**Score per `rubric.md`.** Read each conversation, then write your scores in `human_labels_to_fill.csv` (matching by `id`).",
          "",
          "Markers: answer_copying, no_elaboration, no_error_correction, no_questioning, verbatim_reuse.",
          "",
          "Scores: 0 = absent, 1 = mild, 2 = strong, or `NA` (only allowed for no_error_correction and verbatim_reuse — see rubric).",
          "",
          "---",
          ""]
    for cid, agg, bin_label in shuffled:
        convo = pool[cid]
        turns = convo.get("turns", [])
        n_turns = len(turns)
        md.append(f"## Conversation `{cid}`  *(n_turns={n_turns})*")
        md.append("")
        for i, t in enumerate(turns):
            role = (t.get("role") or "").upper()
            content = (t.get("content") or "").strip()
            # Truncate very long content to keep the file readable
            if len(content) > 2000:
                content = content[:2000] + "\n*[... truncated for readability — full text in conversation_pool.jsonl ...]*"
            md.append(f"### Turn {i+1} — **{role}**")
            md.append("")
            md.append(content)
            md.append("")
        md.append("---")
        md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")

    # CSV template — same shuffled order as the markdown, so you can fill it
    # top-to-bottom while reading the conversations.
    csv_lines = ["id," + ",".join(MARKERS) + ",notes"]
    for cid, _, _ in shuffled:
        csv_lines.append(f"{cid},,,,,,")
    OUT_CSV.write_text("\n".join(csv_lines) + "\n")
    print(f"Wrote {OUT_CSV}")

    # Metadata for reproducibility
    meta = {
        "seed": SEED,
        "n_per_bin": N_PER_BIN,
        "stratified_by": "anthropic__claude-haiku-4-5 aggregate score",
        "n_pool": len(pool),
        "n_anthro_graded": len(anthro),
        "bins": {"low": [c for c, _, b in sampled if b == "low"],
                 "mid": [c for c, _, b in sampled if b == "mid"],
                 "high": [c for c, _, b in sampled if b == "high"]},
        "sample_with_anthro_aggregate": [
            {"id": cid, "anthro_aggregate": agg, "bin": bin_}
            for cid, agg, bin_ in sampled
        ],
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_META}")


if __name__ == "__main__":
    main()
