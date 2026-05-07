"""LLM-as-judge for cognitive offloading markers.

Loads the rubric from rubric.md, formats a conversation, calls an LLM with
strict-JSON instructions, and returns per-marker scores with justifications.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RUBRIC_PATH = Path(__file__).parent / "rubric.md"

MARKERS = [
    "answer_copying",
    "no_elaboration",
    "no_error_correction",
    "no_questioning",
    "verbatim_reuse",
]

GRADER_PROMPT_TEMPLATE = """You are a behavioral coder scoring user turns in human-AI conversations for cognitive offloading markers. Apply the rubric below strictly. You score the USER's behavior across the conversation, not the assistant's.

# Rubric

{rubric}

# Conversation

{conversation}

# Task

For each of the five markers, return an integer score per the rubric anchors, or the string "NA" if the rubric explicitly allows NA for that marker and the conversation gives no basis to score it. Provide a brief justification (<=20 words) per marker, citing concrete behavior from the conversation.

Return JSON exactly in this schema, with no surrounding text:

{{
  "answer_copying": {{"score": 0, "justification": "..."}},
  "no_elaboration": {{"score": 0, "justification": "..."}},
  "no_error_correction": {{"score": 0, "justification": "..."}},
  "no_questioning": {{"score": 0, "justification": "..."}},
  "verbatim_reuse": {{"score": 0, "justification": "..."}}
}}
"""


@dataclass
class GraderConfig:
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 800
    temperature: float = 0.0


def load_rubric() -> str:
    return RUBRIC_PATH.read_text()


def format_conversation(turns: list[dict], max_assistant_chars: int = 2000) -> str:
    """Render a list of {role, content} turns into a flat string for the grader.

    Truncates long assistant turns to keep prompts manageable. Skips turns with
    missing role or content rather than raising.
    """
    if not turns:
        raise ValueError("Cannot grade an empty conversation.")
    out = []
    for i, t in enumerate(turns):
        role = (t.get("role") or "").strip().upper()
        content = (t.get("content") or "").strip()
        if not role or not content:
            continue
        if role == "ASSISTANT" and len(content) > max_assistant_chars:
            content = content[:max_assistant_chars] + "\n[... assistant response truncated ...]"
        out.append(f"[TURN {i+1} - {role}]\n{content}")
    if not out:
        raise ValueError("Conversation contained no usable turns (all empty or malformed).")
    return "\n\n".join(out)


def grade_anthropic(prompt: str, cfg: GraderConfig) -> tuple[dict, str]:
    """Returns (parsed_scores, raw_response_text)."""
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to your .env or environment.")
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    if not resp.content or not hasattr(resp.content[0], "text"):
        raise RuntimeError(f"Unexpected Anthropic response shape: {resp!r}")
    text = resp.content[0].text
    return _extract_json(text), text


def grade_openai(prompt: str, cfg: GraderConfig) -> tuple[dict, str]:
    """Returns (parsed_scores, raw_response_text)."""
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env or environment.")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned empty content.")
    return _extract_json(content), content


def _extract_json(text: str) -> dict:
    """Tolerant JSON extractor.

    Handles: markdown code fences, leading/trailing prose, embedded braces in
    JSON-string values. Uses json.JSONDecoder.raw_decode rather than
    brace-counting (which would miscount braces inside strings).
    """
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        end_fence = text.find("```", 3)
        if end_fence != -1:
            inner = text[3:end_fence]
            # Drop a language tag on the first line ("json", "JSON", etc.)
            if "\n" in inner:
                first_line, rest = inner.split("\n", 1)
                if not first_line.strip().startswith("{"):
                    inner = rest
            text = inner.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found. First 300 chars:\n{text[:300]!r}")

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON parse failed at offset {e.pos}: {e.msg}. "
            f"Text near error: {text[start:start+300]!r}"
        ) from e
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def grade_conversation(turns: list[dict], cfg: Optional[GraderConfig] = None) -> dict:
    """Grade a single conversation.

    Returns a dict with:
      - scores: per-marker {score, justification}
      - raw_response: the full untouched model output (for re-parsing without re-querying)
      - prompt: the full prompt sent (for reproducibility)
      - model, provider: which API was used
    """
    cfg = cfg or GraderConfig()
    rubric = load_rubric()
    convo = format_conversation(turns)
    prompt = GRADER_PROMPT_TEMPLATE.format(rubric=rubric, conversation=convo)

    if cfg.provider == "anthropic":
        scores, raw = grade_anthropic(prompt, cfg)
    elif cfg.provider == "openai":
        scores, raw = grade_openai(prompt, cfg)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider!r}. Use 'anthropic' or 'openai'.")

    _validate_scores(scores)
    return {
        "scores": scores,
        "raw_response": raw,
        "prompt": prompt,
        "model": cfg.model,
        "provider": cfg.provider,
    }


def _validate_scores(scores: dict) -> None:
    """Validate and normalize marker scores in-place.

    Coerces stringified ints ("0", "1", "2") to int, "n/a"/"na" variants to "NA".
    Raises ValueError on missing markers, malformed entries, or invalid scores.
    """
    for m in MARKERS:
        if m not in scores:
            raise ValueError(f"Missing marker {m!r}. Got keys: {list(scores)}")
        entry = scores[m]
        if not isinstance(entry, dict) or "score" not in entry:
            raise ValueError(f"Marker {m!r} has malformed entry: {entry!r}")
        s = entry["score"]
        if isinstance(s, str):
            s_norm = s.strip().upper()
            if s_norm in ("NA", "N/A", "NULL", "NONE"):
                entry["score"] = "NA"
            else:
                try:
                    entry["score"] = int(s_norm)
                except ValueError as e:
                    raise ValueError(f"Invalid score for {m!r}: {s!r}") from e
        if entry["score"] not in (0, 1, 2, "NA"):
            raise ValueError(f"Invalid score for {m!r}: {entry['score']!r}")
        # Justification is optional but normalize to string if present
        if "justification" in entry and entry["justification"] is not None:
            entry["justification"] = str(entry["justification"])


def aggregate_score(scores_or_result: dict) -> float:
    """Mean of applicable markers, range [0, 2]. NaN if all NA.

    Accepts either the inner per-marker dict, or the full grade_conversation
    result dict (with a top-level 'scores' key).
    """
    if "scores" in scores_or_result and isinstance(scores_or_result["scores"], dict) \
            and "answer_copying" in scores_or_result["scores"]:
        scores = scores_or_result["scores"]
    else:
        scores = scores_or_result
    vals = []
    for m in MARKERS:
        s = scores[m]["score"]
        if s == "NA":
            continue
        vals.append(int(s))
    return sum(vals) / len(vals) if vals else float("nan")


if __name__ == "__main__":
    # Quick smoke test on a single example
    test = [
        {"role": "user", "content": "Write a 300-word essay on the French Revolution."},
        {"role": "assistant", "content": "[essay text about French Revolution]"},
        {"role": "user", "content": "Now do one on the American Revolution."},
    ]
    cfg = GraderConfig(provider=os.environ.get("PROVIDER", "anthropic"))
    print(f"Grading test conversation with {cfg.provider}/{cfg.model}...")
    result = grade_conversation(test, cfg)
    print(json.dumps(result["scores"], indent=2))
    print(f"\nAggregate offloading score: {aggregate_score(result['scores']):.2f}")
    print(f"Raw response length: {len(result['raw_response'])} chars")
