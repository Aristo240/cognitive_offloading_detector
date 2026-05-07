# Cognitive Offloading Detector

> **Status: v0.1 prototype.** Exploratory work in progress. Not a peer-reviewed evaluation; no comprehensive literature review yet. Pointers to overlapping prior work are welcome.

A small LLM-as-judge for surfacing behavioral signals related to **cognitive offloading** in human-LLM conversations — five markers visible in dialogue (answer-copying, no-elaboration, no-error-correction, no-questioning, verbatim-reuse) intended to complement, not replace, engagement and outcome measures.

## Why this exists

Engagement, satisfaction, and task-completion metrics describe whether users keep using a product, but they don't directly describe how users engage cognitively with model outputs. This project explores whether a transparent rubric and a simple LLM grader can produce a useful behavioral signal for that question. The aim is a starting point for discussion and iteration, not a settled measurement instrument.

## What it does

1. **Rubric** (`rubric.md`): five behavioral markers proposed as offloading-shaped signals in dialogue (answer-copying, no-elaboration, no-error-correction, no-questioning, verbatim-reuse).
2. **LLM grader** (`grader.py`): scores any conversation against the rubric and returns structured JSON with justifications.
3. **Experiment runner** (`run_experiment.py`): runs the grader over a dataset (synthetic, LMSYS, OASST1, etc.). Saves the full conversation, parsed scores, raw model response, and prompt to JSONL — so re-analysis or re-grading does not require re-querying anything. Resumes interrupted runs automatically.
4. **Re-grader** (`regrade.py`): re-scores a saved JSONL with a different model or after rubric edits, without re-querying the source dataset. Can also re-parse saved raw responses without spending API tokens.
5. **Validation** (`validate.py`): computes Cohen's quadratic-weighted kappa between the LLM grader and human hand-labels.
6. **Analysis** (`analyze.py`): base rates, distribution plots, summary stats.

## Setup

```bash
cd cognitive_offloading_detector
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env  # then fill in your API key
```

Default provider is Anthropic (`claude-haiku-4-5`). Switch with `--provider openai --model gpt-4o-mini`.

## One-day pipeline

```bash
# 1. Run grader on bundled synthetic examples (~$0.05, ~2 min)
python run_experiment.py --source synthetic --n 8 --out results/grades_synthetic.jsonl

# 2. Run on real public chatlogs (~$1-3 per 100 conversations on Haiku)
python run_experiment.py --source hf --data lmsys/lmsys-chat-1m --n 200 --out results/grades_real.jsonl

# 3. Hand-label 30 conversations for validation
#    Open validation/human_labels_template.csv, fill in scores, save as human_labels.csv

# 4. Compute inter-rater agreement
python validate.py --llm results/grades_real.jsonl --human validation/human_labels.csv

# 5. Plot distributions and summary stats
python analyze.py --in results/grades_real.jsonl --outdir results/
```

## Outputs

- `results/grades_*.jsonl`: per-conversation scores with justifications.
- `results/grades.csv`: flat table for analysis.
- `results/distributions.png`: histogram + per-marker means.
- `results/validation.json`: Cohen's kappa per marker.

## Project layout

```
cognitive_offloading_detector/
├── rubric.md                    # The construct definition + scoring anchors
├── grader.py                    # LLM-as-judge implementation
├── run_experiment.py            # Batch grading
├── validate.py                  # Human-LLM agreement
├── analyze.py                   # Stats + plots
├── data/
│   └── synthetic_examples.json  # 8 hand-crafted demo conversations
├── validation/
│   └── human_labels_template.csv
├── results/                     # outputs (generated)
└── writeup_draft.md             # blog post draft
```

## Limitations

- **Single-conversation behavioral signal, not a learning outcome.** Offloading in one conversation does not entail failure to learn; the rubric describes observed dialogue behavior, not capability change.
- **Behavior, not intent.** The rubric scores observable conversational behavior. Some offloading is appropriate (e.g., boilerplate). The rubric does not distinguish.
- **Construct validity is not established.** Whether the five markers track a coherent underlying construct is an empirical question that has not been tested here.
- **Inter-rater reliability is not established.** v0.1 reports agreement against a single annotator's hand-labels at most. Multi-rater validation is needed before treating this as a measurement instrument.
- **English-language conversations only.**
- **No comprehensive literature review yet.** Related constructs and instruments may already exist in cognitive science, HCI, education research, or AI evaluation; pointers welcome.

## Citation

If this is useful as a starting point: "Rozen, N. (2026). Cognitive Offloading Detector v0.1. [github URL]". Treat as a prototype, not a validated instrument.
