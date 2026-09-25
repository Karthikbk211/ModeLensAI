"""
GenAI layer: turns the structured failure-analysis output into a plain-
English explanation, possible contributing factors, and suggested next
steps -- deliberately phrased as hypotheses, not diagnoses, since the
system observes error patterns but can't prove root cause.

Uses the Google Gemini API (gemini-3.8-flash, free tier). Requires
GEMINI_API_KEY to be set in the environment or Streamlit secrets.
If it's missing, `explain_failures` returns a clearly-labeled fallback
so the rest of the app still works.
"""

from __future__ import annotations

import json
import re
import time

from google import genai
from google.genai import types

from src.utils.preprocessing import get_config

CANDIDATE_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-pro",
]
MAX_RETRIES_PER_MODEL = 2
INITIAL_BACKOFF_SEC = 1.5
BACKOFF_FACTOR = 2.0

SYSTEM_PROMPT = """You are an assistant embedded in ModelLens AI, a tool that analyzes \
where machine learning classification models fail. You are given the *already computed* \
metrics and failure-analysis output for one evaluation run -- you do not see the raw data \
or model yourself, and you cannot run further experiments.

Write a failure report with three parts:
1. "summary": 2-4 sentences describing the model's main weakness in plain English, \
grounded strictly in the measured findings given and, where useful, the retrieved knowledge context.
2. "contributing_factors": 3-5 short bullet-style strings naming *possible* explanations \
(e.g. class imbalance, feature overlap, insufficient data for a class). Use hedged \
language ("may", "could", "is consistent with") -- never state a cause as certain, since \
you cannot prove causation from these numbers alone. Retrieved knowledge is guidance, not evidence that a specific cause is true.
3. "recommendations": 3-5 short, concrete, actionable next steps someone could take to \
investigate or improve the model.

Respond with ONLY a JSON object with exactly these three keys: "summary" (string), \
"contributing_factors" (array of strings), "recommendations" (array of strings). \
No markdown, no preamble, no code fences."""


def _build_user_prompt(metrics: dict, failure_analysis: dict, model_name: str, n_samples: int, retrieved_context: list[dict] | None = None) -> str:
    payload = {
        "model_name": model_name,
        "n_samples": n_samples,
        "metrics": metrics,
        "class_failures": failure_analysis.get("class_failures", []),
        "top_misclassifications": failure_analysis.get("top_misclassifications", []),
        "feature_errors": failure_analysis.get("feature_errors", []),
        "retrieved_knowledge": retrieved_context or [],
    }
    return json.dumps(payload, indent=2)


def _fallback_result(reason: str) -> dict:
    return {
        "summary": f"GenAI explanation unavailable ({reason}). "
        "The numeric failure analysis above is still complete and accurate -- "
        "set GEMINI_API_KEY to enable automatic explanations.",
        "contributing_factors": [],
        "recommendations": [],
    }


def _extract_json(text: str) -> dict:
    """Robustly extract a JSON object from model output that may contain
    markdown fences or surrounding prose."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("No JSON object found", text, 0)


def explain_failures(
    metrics: dict, failure_analysis: dict, model_name: str = "model", n_samples: int = 0, retrieved_context: list[dict] | None = None
) -> dict:
    """Call the Google Gemini API with the structured analysis and return
    {summary, contributing_factors, recommendations}. Automatically retries
    and falls back across stable Gemini models on transient server/quota errors."""
    api_key = get_config("GEMINI_API_KEY")
    if not api_key:
        return _fallback_result("no GEMINI_API_KEY set")

    client = genai.Client(api_key=api_key)
    user_prompt = _build_user_prompt(metrics, failure_analysis, model_name, n_samples, retrieved_context=retrieved_context)

    last_exception = None
    for model in CANDIDATE_MODELS:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        max_output_tokens=1000,
                        temperature=0.3,
                    ),
                )
                text = response.text or ""
                parsed = _extract_json(text)
                return {
                    "summary": parsed.get("summary", ""),
                    "contributing_factors": parsed.get("contributing_factors", []),
                    "recommendations": parsed.get("recommendations", []),
                }
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                exc_str = str(exc)
                is_transient = any(
                    keyword in exc_str
                    for keyword in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "500", "502", "504", "high demand", "NOT_FOUND", "404")
                )
                if is_transient and attempt < MAX_RETRIES_PER_MODEL - 1:
                    sleep_sec = INITIAL_BACKOFF_SEC * (BACKOFF_FACTOR ** attempt)
                    time.sleep(sleep_sec)
                    continue
                # If non-retryable or retries exhausted for this model, try the next model in CANDIDATE_MODELS
                break

    return _fallback_result(f"API error: {last_exception}")


