# Cognitive Offloading Markers in Human-LLM Conversations

## Construct

**Cognitive offloading** in human-LLM dialogue: the user delegates the cognitive work of a task to the AI rather than using AI assistance to enhance their own reasoning. This rubric scores **user turns** (not AI responses) on five behavioral markers visible in dialogue.

This is a **behavioral measure**, not a measure of intent or of long-term learning outcomes. It tells you what a user *did*, not what they *understood*.

## Markers

### 1. Answer-Copying (AC)
**Definition.** User asks for a final answer or completed artifact rather than help understanding or working through the problem.

- **0 (absent)**: User asks for explanation, hints, partial help, verification of own work, or scaffolding.
- **1 (mild)**: User asks for an answer but with some context suggesting they tried first ("I got X, is that right?", "I'm stuck after step 3, can you continue?").
- **2 (strong)**: User requests complete solutions with no engagement signals ("write me a 500-word essay on X", "give me the answer to question 3", "do my homework").

### 2. No-Elaboration (NE)
**Definition.** User accepts the AI's response without expanding, restating in own words, or building on it.

- **0 (absent)**: User builds on, restates, or applies the AI's output ("OK so the key idea is X — does that mean Y?").
- **1 (mild)**: User acknowledges briefly ("thanks", "ok") then changes topic.
- **2 (strong)**: User accepts and immediately requests the next deliverable without engagement.

### 3. No-Error-Correction (NEC)
**Definition.** When the AI's response contains an error, ambiguity, hallucination, or invitation to clarify, the user does not push back, correct, or probe.

- **0 (absent)**: User catches errors, corrects, or asks for clarification on something dubious.
- **1 (mild)**: User notices something is off but does not investigate.
- **2 (strong)**: AI response contains a notable issue and user accepts it uncritically and proceeds.
- **NA**: AI responses contain no apparent issues to correct.

### 4. No-Questioning (NQ)
**Definition.** User does not ask follow-up questions that probe reasoning, mechanism, or alternatives.

- **0 (absent)**: User asks "why", "how does that work", "what about X", or otherwise probes the AI's reasoning.
- **1 (mild)**: User asks only logistical/format questions ("can you reformat", "in bullet points please").
- **2 (strong)**: User asks only for more outputs, never for understanding.

### 5. Verbatim-Reuse (VR)
**Definition.** Whether the user reuses AI output verbatim. This can be evidenced either (a) by the user producing text in subsequent turns that is substantially copied from prior AI output, or (b) by the user explicitly stating intent to use AI output verbatim externally ("I'll use that exact paragraph", "thanks, copying this into my essay").

- **0 (absent)**: User text is independent or paraphrases the AI's contribution; no statement of verbatim reuse.
- **1 (mild)**: User text borrows phrases but is mostly own; or vague statements of reuse.
- **2 (strong)**: User text is largely identical to AI output; or explicit statement of verbatim-reuse intent.
- **NA**: User does not produce comparable own-text and makes no statement about reuse.

## Aggregate score

`offloading_score = mean(applicable markers)`, range [0, 2]:

- **0.0–0.5**: thinking-with-AI
- **0.5–1.0**: mixed
- **1.0–2.0**: offloading-dominant

Markers scored "NA" are excluded from the mean.

## Scope and limitations

This rubric does NOT measure:
- **Intent.** A user copying an essay may be cheating, may be learning by example, or may be paid to produce that essay for someone else. We score behavior.
- **Appropriateness.** Offloading boilerplate (formatting, syntax) is fine; offloading reasoning is the construct of concern. The rubric does not distinguish — context does.
- **Long-term learning.** This is a within-conversation signal. Persistent offloading across many tasks is what should worry us; isolated offloading is normal.
- **Causal effects on capability.** Detecting offloading is a prerequisite for studying its effects, not a substitute.

## Possible use cases

- **Conversation-level descriptive analysis** of how AI products are used.
- **Comparative metrics** for offloading rates across user segments, products, or model versions, treated as descriptive signals rather than diagnoses.
- **Candidate moderator** in learning-effectiveness studies — empirical question whether offloading rate relates to transfer-test performance.
- **Behavioral input** to broader investigations of AI use in learning settings.

These are framings the rubric is intended to support; whether it does so reliably in any given setting is an empirical question that requires validation in that setting.

## Related work and prior art

"Cognitive offloading" is an established construct in cognitive psychology (e.g., Risko & Gilbert, 2016; Storm & Stone, 2015). The novelty claim of this artifact is narrow: a small, transparent rubric and LLM-based grader for surfacing offloading-shaped behavior in human-LLM dialogue specifically. Closely related work likely exists in HCI, education research, and AI evaluation; this project does not claim priority and welcomes pointers to overlapping literature.
