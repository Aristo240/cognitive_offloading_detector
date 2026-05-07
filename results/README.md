# Results layout

Each subdirectory is one cross-judge run. All directories share the same file
schema (one JSONL per judge, plus a shared `conversation_pool.jsonl`,
`agreement.json`, and `run_metadata.json`).

## Runs

| Dir | Source | n | Judges | Notes |
|---|---|---:|---|---|
| `cross_judge_synthetic/` | Hand-crafted contrastive examples (`data/synthetic_examples.json`) | 8 | Anthropic Haiku + OpenAI gpt-4o-mini + Gemini 2.0 Flash | Sanity check. Inter-judge aggregate r ≈ 0.93–0.96. |
| `cross_judge_wildchat/` | `allenai/WildChat-1M` (HuggingFace) | 100 | Anthropic Haiku + Gemini 2.0 Flash | OpenAI excluded — WildChat is GPT-generated (within-family confound). Median 2 turns; agreement weak (r ≈ 0.26). |
| `cross_judge_ultrachat/` | `HuggingFaceH4/ultrachat_200k` (HuggingFace) | 100 | Anthropic Haiku + Gemini 2.0 Flash + (3rd judge: Llama-3.3-70B-Instruct-FP8 self-hosted on Lambda H100) | OpenAI excluded — UltraChat is GPT-3.5/4 generated. Llama judge added for out-of-family triangulation. |

## Per-run files

- `conversation_pool.jsonl` — the exact conversations all judges saw, including
  source `model` field where available.
- `<provider>__<model>.jsonl` — one row per (conversation, judge), with the
  judge's full per-marker scores, justifications, raw model response, prompt,
  and timestamp.
- `agreement.json` — pairwise inter-judge metrics: quadratic-weighted Cohen's
  kappa per marker + Pearson r on aggregate scores.
- `run_metadata.json` — exact configuration of this run.

For some runs:

- `distributions.png` — histogram of aggregate scores + per-marker means
  (output of `analyze.py`).
- `grades.csv` — flat per-conversation table.

## Open-weights 3rd judge: landed (UltraChat)

A 3rd judge from a 4th family — **Llama-3.3-70B-Instruct-FP8** self-hosted on
a Lambda Cloud H100 PCIe (80 GB) via vLLM in the official `vllm/vllm-openai`
Docker image — graded **92/100** UltraChat conversations (8 failed: long-conversation
token-budget edge cases at `--max-model-len 4096`, handled gracefully by the
grader's resumability-friendly write path).

Per-judge files for this run:
- `cross_judge_ultrachat/anthropic__claude-haiku-4-5.jsonl` (n=100)
- `cross_judge_ultrachat/gemini__gemini-2.0-flash.jsonl` (n=100)
- `cross_judge_ultrachat/vllm__Llama-3.3-70B-Instruct-FP8.jsonl` (n=92)
- `cross_judge_ultrachat/llama/` — distribution plot + grades.csv for Llama only

### 3-judge cross-family agreement (UltraChat n=92–100)

| Pair | n | Aggregate Pearson r | Best per-marker κ |
|---|---:|---:|---|
| Anthropic Haiku ↔ **Llama-3.3-70B-FP8** | 92 | **0.40** | answer_copying κ=0.47 |
| Anthropic Haiku ↔ Gemini Flash | 100 | 0.17 | answer_copying κ=0.04 |
| Gemini Flash ↔ Llama-3.3-70B-FP8 | 92 | 0.18 | no_questioning κ=0.12 |

**Reading.** Anthropic and Llama-70B (different families, different training
pipelines) converge moderately; Gemini diverges from both. The Anthropic-vs-Gemini
disagreement on UltraChat is therefore better explained by **judge calibration drift
(Gemini being systematically conservative — 98% of conversations rated
thinking-with-AI)** than by genuine rubric ambiguity. Hand-coded ground truth
(see `validation/`) will pin down which judges are closest to human labels.

Path to land it (recorded for reproducibility): the official `vllm/vllm-openai`
Docker image (skipping the entire pip dep stack), `--max-model-len 4096` to leave
headroom for KV cache after the 70 GB FP8 weights, prequantized
`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` weights. Setup script:
`serve_and_judge_docker.sh`.
