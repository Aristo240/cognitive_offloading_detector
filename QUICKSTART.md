# Quickstart — get this shipping in one day

Total: ~5–8 hours of work. Most of the time is in steps 4 (real data) and 5 (hand-labeling).

## 0. Setup (15 min)

```bash
cd "cognitive_offloading_detector"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env
```

Edit `.env` and add your `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` if using `--provider openai`).

## 1. Smoke test on one example (5 min, ~$0.001)

```bash
python grader.py
```

You should see JSON output with five marker scores and an aggregate. If this fails, fix API auth before going further.

## 2. Run on the 8 bundled synthetic examples (5 min, ~$0.05)

```bash
python run_experiment.py --source synthetic --n 8 --out results/grades_synthetic.jsonl
python analyze.py --in results/grades_synthetic.jsonl --outdir results/
```

Sanity check: examples labeled "high offloading" in the JSON should score ≥ 1.0 aggregate; "low offloading" examples should score ≤ 0.5. If not, the grader prompt or rubric needs tightening.

## 3. Inspect a few outputs by eye (15 min)

```bash
cat results/grades_synthetic.jsonl | python -m json.tool
```

Read the justifications. Are they citing concrete behavior? Are any obviously wrong? This is your first iteration on rubric clarity.

## 4. Run on a real public dataset (1–2 hours, ~$2–5)

LMSYS-Chat-1M is one option. Requires HuggingFace authentication.

```bash
huggingface-cli login  # one-time
python run_experiment.py --source hf --data lmsys/lmsys-chat-1m --n 200 --out results/grades_real.jsonl
python analyze.py --in results/grades_real.jsonl --outdir results/
```

If LMSYS is gated for you, try `OpenAssistant/oasst1` (open) — the conversation format is tree-structured so `_extract_turns` may need adjustment.

**Resumability and raw-data persistence.** If the run is interrupted, simply re-run the same command — already-graded IDs are skipped. The conversations themselves are also cached separately to `results/grades_real_conversations.jsonl`, so you never have to re-query HuggingFace. Each output row contains the full `turns`, `scores`, `raw_response`, and `prompt`, so re-analysis or re-grading does not require re-querying anything.

**Re-grading without spending more API tokens.** If you tweak `rubric.md` and want to re-score:

```bash
python regrade.py --in results/grades_real.jsonl --out results/grades_real_v2.jsonl
```

Or to re-parse saved raw responses (no API calls at all):

```bash
python regrade.py --in results/grades_real.jsonl --out results/grades_reparsed.jsonl --reparse-only
```

## 5. Hand-label 30 conversations for validation (2–3 hours)

This is the highest-effort step but also the highest-credibility one. Most LLM-as-judge papers skip it.

```bash
# Pick 30 conversations from results/grades_real.jsonl with diverse aggregate scores
# Open validation/human_labels_template.csv in a spreadsheet
# Read each conversation, score per rubric.md, save as validation/human_labels.csv
python validate.py --llm results/grades_real.jsonl --human validation/human_labels.csv
```

See `validation/labeling_instructions.md` for the protocol.

**If you genuinely cannot label 30**: do 15. Document n. Caveat clearly in the writeup.

## 6. Fill the writeup template (30–60 min)

Open `writeup_draft.md`. Replace every `[X]`, `[N]`, `[dataset]`, `[INTERPRET]`, `[INSERT]` with your actual values and a sentence of interpretation. Drop in `results/distributions.png`.

## 7. Ship (30 min)

- Push to GitHub. Make sure `.env` and `results/*.jsonl` (if they contain raw real data) are gitignored.
- Post to LinkedIn / X with a one-paragraph summary and the figure.
- Add to your CV under "Independent AI Safety Research Projects":
  > **Cognitive Offloading in LLM Conversations**: built and validated an LLM-as-judge for detecting cognitive offloading markers in human-AI dialogue (5-marker rubric, validated against hand-labels on 30 conversations, Cohen's κ = X). Repository includes rubric, grader, validation set, and analysis. [github]

## Cost ceiling

With Claude Haiku 4.5: roughly $0.005–$0.02 per conversation depending on length. 200 conversations ≈ $1–$4. Hand-labeling is the binding constraint, not API cost.

## If something breaks

| Failure | Likely cause | Fix |
|---|---|---|
| `ValueError: No JSON object found` | Model wrapped JSON in prose | Re-run; lower temperature; for OpenAI use `response_format={"type": "json_object"}` (already on) |
| `KeyError: 'role'` | HF dataset uses different schema | Edit `_extract_turns` in `run_experiment.py` |
| `403` on HF | Dataset is gated | `huggingface-cli login` or pick an open dataset |
| All scores cluster at 0 | Conversations are too short or atypical | Increase `--min-turns` filter or sample differently |
| Kappa is `null` | All your scores are the same | Re-sample conversations with more behavioral variety |
