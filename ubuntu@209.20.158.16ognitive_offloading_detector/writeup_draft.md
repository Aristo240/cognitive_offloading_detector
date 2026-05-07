# Detecting cognitive offloading in LLM conversations: a small prototype

*Naama Rozen — [date]. v0.1 prototype, not a validated instrument.*

*Repo: [github.com/...]*

---

## TL;DR

I built a small LLM-as-judge that scores user turns in human-AI conversations on five candidate behavioral markers of cognitive offloading: **answer-copying, no-elaboration, no-error-correction, no-questioning, verbatim-reuse**. On `[N]` conversations from `[dataset]`, agreement against my own hand-labels was quadratic-weighted Cohen's κ = `[VALUE]` (single annotator, so this is a ceiling on reliability, not an estimate of it). Base rates of strong markers varied between `[X]%` and `[Y]%`. This is a starting point for discussion, not a settled measurement instrument; pointers to overlapping prior work in HCI, education research, and AI evaluation are welcome.

---

## Why this exists

Engagement, satisfaction, retention, and task-completion metrics describe whether users keep using a product, but they describe less about *how* users engage cognitively with model outputs within a session.

The same conversation can produce similar engagement metrics whether the user is paraphrasing, probing, and correcting errors, or asking for finished artifacts and accepting outputs uncritically. Distinguishing these patterns at scale would be useful — both for descriptive analysis of how AI products are used, and as one possible input to studies on the cognitive effects of AI assistance.

This project explores whether a small, transparent rubric and an LLM-based grader can produce a behavioral signal aimed at that distinction. It does not aim to solve measurement of learning outcomes; it aims to provide a cheap conversation-level behavioral artifact that researchers can criticize, extend, or replace.

## Prior work and scope

"Cognitive offloading" is a longstanding construct in cognitive psychology (Risko & Gilbert, 2016; Storm & Stone, 2015; Sparrow et al., 2011). Closely related work likely exists in HCI on AI-assisted task delegation, in education research on help-seeking and answer-copying, and in AI evaluation on grader rubrics. I have not done a comprehensive literature review for this prototype. The contribution here is narrow: a small reusable rubric, a transparent grader, and saved raw outputs that anyone can inspect or re-grade.

## The construct

Cognitive offloading, in this rubric, is the user delegating the cognitive work of a task to the AI rather than using AI assistance to enhance their own reasoning. The rubric scores **user turns** (not AI responses) on five markers visible in dialogue:

1. **Answer-copying** - asks for finished outputs vs. for help understanding.
2. **No-elaboration** - accepts AI output without paraphrasing, applying, or building on it.
3. **No-error-correction** - does not push back when the AI is wrong, ambiguous, or sycophantic.
4. **No-questioning** - asks no follow-ups that probe reasoning, mechanism, or alternatives.
5. **Verbatim-reuse** - in subsequent text, copies AI output rather than producing own.

Each marker is scored 0/1/2 with concrete anchors (or NA for markers that don't apply). The aggregate is the mean of applicable markers, range [0, 2].

This is **a behavioral measure**, not a measure of intent or of long-term learning outcomes. Some offloading is appropriate (boilerplate, formatting). The construct of concern is offloading *reasoning*, not offloading *typing*. Persistent offloading across many tasks is what should worry us; isolated offloading is normal.

## The grader

A standard LLM-as-judge: rubric in the system prompt, conversation in the user message, JSON-only output with per-marker scores and short justifications. Default model is Claude Haiku 4.5 (cheap, capable of structured output). Code in `grader.py`.

## Validation

I hand-labeled `[N=30]` conversations sampled from `[dataset]`, blind to the LLM's grades. Inter-rater agreement (LLM vs. human, quadratic-weighted Cohen's κ):

| Marker | n | κ (quadratic) | exact agreement |
|---|---:|---:|---:|
| Answer-copying | `[X]` | `[X]` | `[X]` |
| No-elaboration | `[X]` | `[X]` | `[X]` |
| No-error-correction | `[X]` | `[X]` | `[X]` |
| No-questioning | `[X]` | `[X]` | `[X]` |
| Verbatim-reuse | `[X]` | `[X]` | `[X]` |

[INTERPRET: which markers showed higher/lower agreement, and one or two sentences of plausible reasons. Be careful not to overgeneralize from a single annotator + single dataset.]

## Findings on `[dataset]`

[INSERT: distribution figure - aggregate offloading score histogram + per-marker means.]

Among `[N]` conversations, the aggregate offloading score had mean `[X]`, median `[Y]`. The distribution split as:

- Thinking-with-AI (score < 0.5): `[X]%`
- Mixed (0.5–1.0): `[X]%`
- Offloading-dominant (≥ 1.0): `[X]%`

[OPTIONAL: break down by conversation type - coding vs. writing vs. open-ended Q&A. Coding conversations likely show high error-correction; essay-writing conversations likely show high answer-copying and verbatim-reuse.]

## What this is not

- **Not a learning-outcome measure.** Within-conversation behavior is not a capability assessment. A user can offload one task and learn effectively elsewhere.
- **Not a judgment about users.** Some uses of AI for cognitive offloading are appropriate (drafting boilerplate, formatting). The rubric is descriptive.
- **Not validated for non-English conversations.**
- **Not reliability-validated.** v0.1 uses single-rater hand-labels; multi-rater validation has not been done.
- **Not novel as a construct.** Cognitive offloading is well-established in cognitive psychology. The novelty claim here is narrow: a small open rubric and grader for LLM dialogue specifically.

## Possible uses

If the markers turn out to be reliable in a given setting (an empirical question), a grader like this could plausibly support:

1. **Descriptive analysis** of how users engage with deployed AI products, complementing CSAT-style metrics.
2. **Comparative monitoring** across model versions, user segments, or product surfaces.
3. **Candidate moderator** in learning-effectiveness studies — whether offloading rate predicts transfer-test performance is testable.
4. **Behavioral input** alongside other measures in cognitive-effects research.

These are framings the rubric is intended to support, not claims about what it has been shown to do.

## Limitations and next steps

- Literature review against the established cognitive-offloading and AI-evaluation literature.
- Multi-rater validation with independent annotators.
- Application to multiple datasets (LMSYS, OASST, education-specific corpora) to check whether base rates and reliability are stable.
- Adding a marker (or markers) for *productive* engagement; the current rubric scores what is absent, not what is present.
- Pairing with a transfer-test study to test whether offloading rate during AI-assisted learning relates to post-test performance.

## How to use this

```bash
git clone [repo]
cd cognitive_offloading_detector
pip install -r requirements.txt
cp .env.example .env  # add your API key
python run_experiment.py --source synthetic --n 8 --out results/grades.jsonl
python analyze.py --in results/grades.jsonl --outdir results/
```

The rubric, grader, hand-labels, and analysis scripts are all open. Use, modify, criticize.

---

*Built in one day. Comments and PRs welcome - particularly from anyone working on AI evaluation in education or learning-effects research.*
