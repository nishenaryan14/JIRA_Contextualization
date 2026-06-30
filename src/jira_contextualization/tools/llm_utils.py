"""LLM utility functions for the Jira Contextualization pipeline.

Provides factory functions for creating pre-configured LLM instances
(DeepSeek Chat, DeepSeek Reasoner, Gemini Flash) and helper utilities
for robust LLM interactions including:

- Exponential-backoff retry on transient/rate-limit errors.
- Markdown-fence-aware JSON parsing of LLM responses.
- A convenience wrapper that combines retry + JSON parsing with a
  configurable fallback value.

Usage
-----
>>> from jira_contextualization.tools.llm_utils import get_deepseek_llm, safe_llm_extract
>>> llm = get_deepseek_llm()
>>> result = safe_llm_extract(llm, "Return a JSON object with key 'status': 'ok'")
"""

from __future__ import annotations

import builtins
import json
import os
import time
from functools import partial
from typing import Any, Type

from crewai import LLM
from pydantic import BaseModel

print = partial(builtins.print, flush=True)


# ---------------------------------------------------------------------------
# LLM factory helpers
# ---------------------------------------------------------------------------


def get_deepseek_llm() -> LLM:
    """Return a DeepSeek Chat LLM configured for extraction tasks.

    Uses a low temperature (0.1) for deterministic, factual output and an
    8 000-token response budget suitable for per-issue extraction prompts.
    """
    return LLM(
        model="deepseek/deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.1,
        max_tokens=8000,
    )


def get_deepseek_reasoner_llm() -> LLM:
    """Return a DeepSeek Reasoner LLM for consolidation tasks.

    The Reasoner variant performs deeper chain-of-thought reasoning,
    making it well-suited for cross-issue deduplication and enrichment.
    Temperature is pinned to 0.0 and the token budget is doubled (16 000).
    """
    return LLM(
        model="deepseek/deepseek-reasoner",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0,
        max_tokens=16000,
    )


def get_gemini_llm() -> LLM:
    """Return a Gemini 2.0 Flash LLM for validation tasks.

    Gemini Flash offers fast inference with good instruction-following,
    ideal for structured validation checks.
    """
    return LLM(
        model="gemini/gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
        max_tokens=8000,
    )


# ---------------------------------------------------------------------------
# Retry & parsing utilities
# ---------------------------------------------------------------------------


def llm_call_with_retry(
    llm: LLM,
    messages: list[dict[str, str]],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 3.0,
) -> str:
    """Call an LLM with exponential-backoff retry on transient errors.

    Only rate-limit (429), server (500/503), timeout, and quota errors
    are retried; all other exceptions propagate immediately.

    Args:
        llm: The :class:`crewai.LLM` instance to call.
        messages: Chat-style message list, e.g.
            ``[{"role": "user", "content": "..."}]``.
        max_retries: Maximum number of *retries* (total attempts =
            ``max_retries + 1``).
        initial_delay: Seconds to wait before the first retry.
        backoff_factor: Multiplier applied to the delay after each retry.

    Returns:
        The LLM response as a plain string.

    Raises:
        Exception: Re-raised from the underlying LLM call if all retries
            are exhausted or the error is non-retryable.
    """
    last_error: Exception | None = None
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            response = llm.call(messages)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Only retry on rate limit or transient errors
            is_retryable = any(
                kw in error_str
                for kw in [
                    "429",
                    "rate_limit",
                    "rate limit",
                    "quota",
                    "resource_exhausted",
                    "overloaded",
                    "503",
                    "500",
                    "timeout",
                ]
            )
            if not is_retryable or attempt == max_retries:
                raise
            print(
                f"    ⚠️  LLM call failed (attempt {attempt + 1}/{max_retries + 1}): "
                f"{str(e)[:100]}"
            )
            print(f"    ⏳ Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay *= backoff_factor
    raise last_error  # type: ignore[misc]


def parse_llm_json(text: str) -> dict | list:
    """Parse a JSON object or array from an LLM response string.

    Handles the common case where the model wraps its JSON output in
    Markdown code fences (````json ... ````) by stripping them before
    parsing.

    Args:
        text: Raw LLM response text, potentially wrapped in fences.

    Returns:
        The parsed JSON as a ``dict`` or ``list``.

    Raises:
        json.JSONDecodeError: If the text is not valid JSON after cleanup.
    """
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        # Remove first line (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return json.loads(text)


def safe_llm_extract(
    llm: LLM,
    prompt: str,
    fallback: dict | list | None = None,
    max_retries: int = 3,
) -> dict | list:
    """Call an LLM, parse the JSON response, with retry and fallback.

    This is a convenience wrapper that chains :func:`llm_call_with_retry`
    and :func:`parse_llm_json`, returning *fallback* on any error so that
    upstream pipelines can continue gracefully.

    Args:
        llm: The :class:`crewai.LLM` instance.
        prompt: The user prompt to send.
        fallback: Value to return if the call or parse fails.
            Defaults to an empty ``dict``.
        max_retries: Passed through to :func:`llm_call_with_retry`.

    Returns:
        Parsed JSON (``dict`` or ``list``), or *fallback* on failure.
    """
    if fallback is None:
        fallback = {}
    try:
        raw = llm_call_with_retry(
            llm,
            [{"role": "user", "content": prompt}],
            max_retries=max_retries,
        )
        return parse_llm_json(raw)
    except Exception as e:
        print(f"    ❌ LLM extraction failed: {str(e)[:150]}")
        return fallback
