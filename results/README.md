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

## Open-weights judge: attempted, deferred

A 3rd judge from a 4th family — **Llama-3.3-70B-Instruct-FP8** self-hosted on
a Lambda Cloud H100 PCIe via vLLM — was attempted on the UltraChat pool but
not landed. The Lambda Cloud orchestrator (`lambda_run.py`) and the remote
serving script (`serve_and_judge.sh`) both work; the model downloads cleanly
from HuggingFace. The blocker was a stack of dependency incompatibilities
between Lambda Stack's NVIDIA driver, pip-installed `torch`, `transformers`,
`numpy`, and `vllm`. Each pinned-version fix unlocked a new ABI mismatch
(driver 12.8 vs torch CUDA 13.0; numpy 2.x vs system scipy; transformers
TokenizersBackend vs vllm 0.8.5; `torch_c_dlpack_ext` ABI vs torch 2.6).

This is deferred to v0.2 with a more controlled environment (e.g. a Docker
image with all deps frozen, or Lambda's prebuilt vLLM image when available).
The two-judge UltraChat agreement numbers stand on their own as a
calibration-mismatch finding without a 3rd judge to confirm.
