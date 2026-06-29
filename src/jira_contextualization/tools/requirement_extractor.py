"""Deterministic requirement extractor.

Extracts structured requirement artefacts from a ``NormalizedIssue``
using *only* pattern matching and section-header parsing — **no LLM
calls**.  The AI-powered enrichment stage can build on top of these
deterministic extractions.

Public API
----------
- ``extract_requirements_deterministic(issue)`` → ``dict``
"""

from __future__ import annotations

import re
from typing import Any

from jira_contextualization.models.normalized_issue import NormalizedIssue
from jira_contextualization.tools.wiki_markup_parser import (
    extract_sections,
    parse_jira_markup,
)

# ── Section-heading aliases (case-insensitive matching) ──────────────────
_AC_HEADINGS = {
    "acceptance criteria",
    "ac",
    "acceptance criterion",
    "acceptance",
    "criteria",
    "definition of done",
    "dod",
}
_BUSINESS_RULE_HEADINGS = {
    "business rules",
    "business rule",
    "rules",
    "rule",
    "domain rules",
}
_CONSTRAINT_HEADINGS = {
    "constraints",
    "constraint",
    "limitations",
    "limitation",
    "restrictions",
    "non-functional requirements",
    "nfrs",
}
_USER_STORY_RE = re.compile(
    r"[Aa]s\s+(?:a|an)\s+(.+?),?\s+[Ii]\s+want\s+(?:to\s+)?(.+?)"
    r"(?:,?\s+[Ss]o\s+(?:that\s+)?(.+))?",
    re.DOTALL,
)
_GWT_GIVEN_RE = re.compile(r"\*?Given\*?\s+(.+)", re.IGNORECASE)
_GWT_WHEN_RE = re.compile(r"\*?When\*?\s+(.+)", re.IGNORECASE)
_GWT_THEN_RE = re.compile(r"\*?Then\*?\s+(.+)", re.IGNORECASE)


def extract_requirements_deterministic(issue: NormalizedIssue) -> dict[str, Any]:
    """Deterministically extract requirement artefacts from an issue.

    This function does **not** invoke an LLM — it relies solely on
    regex patterns and Jira wiki-markup section headings.

    Args:
        issue: A parsed ``NormalizedIssue``.

    Returns:
        Dictionary with keys:

        - ``acceptance_criteria`` – ``list[str]``: individual AC items.
        - ``given_when_then``    – ``list[dict]``: structured GWT blocks.
        - ``user_stories``       – ``list[dict]``: parsed user-story parts.
        - ``business_rules``     – ``list[str]``: business-rule items.
        - ``constraints``        – ``list[str]``: constraint items.
    """
    # Parse all text sources
    description_parsed = parse_jira_markup(issue.description) if issue.description else None
    ac_parsed = parse_jira_markup(issue.acceptance_criteria_raw) if issue.acceptance_criteria_raw else None

    # Use pre-parsed sections if available, otherwise derive from raw description
    sections = issue.description_sections or {}
    if not sections and issue.description:
        sections = extract_sections(issue.description)

    ac_sections: dict[str, str] = {}
    if issue.acceptance_criteria_raw:
        ac_sections = extract_sections(issue.acceptance_criteria_raw)

    # ── Acceptance criteria ──────────────────────────────────────────
    acceptance_criteria = _extract_ac_items(sections, ac_sections, description_parsed, ac_parsed)

    # ── Given/When/Then ──────────────────────────────────────────────
    given_when_then = _extract_gwt(description_parsed, ac_parsed)

    # ── User stories ─────────────────────────────────────────────────
    user_stories = _extract_user_stories(issue)

    # ── Business rules ───────────────────────────────────────────────
    business_rules = _extract_by_heading(sections, _BUSINESS_RULE_HEADINGS)

    # ── Constraints ──────────────────────────────────────────────────
    constraints = _extract_by_heading(sections, _CONSTRAINT_HEADINGS)

    return {
        "acceptance_criteria": acceptance_criteria,
        "given_when_then": given_when_then,
        "user_stories": user_stories,
        "business_rules": business_rules,
        "constraints": constraints,
    }


# ── Private helpers ──────────────────────────────────────────────────────


def _extract_ac_items(
    description_sections: dict[str, str],
    ac_sections: dict[str, str],
    description_parsed: dict[str, Any] | None,
    ac_parsed: dict[str, Any] | None,
) -> list[str]:
    """Collect acceptance-criteria items from all available sources.

    Priority order:
    1. Dedicated AC custom field (``acceptance_criteria_raw``).
    2. Sections in the description matching AC-like headings.
    3. Bullet points from the parsed AC field.
    4. Bullet points from description sections matching AC headings.
    """
    items: list[str] = []

    # 1. From the AC field sections
    for heading, body in ac_sections.items():
        items.extend(_lines_from_section(body))

    # If AC field had no sections, use its bullet points
    if not items and ac_parsed:
        items.extend(ac_parsed.get("bullet_points", []))
        # If still nothing, split AC text by newlines
        if not items:
            clean = ac_parsed.get("clean_text", "")
            if clean:
                items.extend(_split_into_items(clean))

    # 2. From description sections with AC-like headings
    for heading, body in description_sections.items():
        if heading.lower().strip() in _AC_HEADINGS or _is_ac_heading(heading):
            items.extend(_lines_from_section(body))

    # 3. Bullet points from description that look like AC
    if not items and description_parsed:
        bullets = description_parsed.get("bullet_points", [])
        # If description has bullets but no AC heading, we include them only
        # if the description doesn't have other structured sections
        if bullets and not description_sections:
            items.extend(bullets)

    # Deduplicate while preserving order
    return _deduplicate(items)


def _extract_gwt(
    description_parsed: dict[str, Any] | None,
    ac_parsed: dict[str, Any] | None,
) -> list[dict[str, str | list[str]]]:
    """Merge GWT blocks from description and AC field."""
    blocks: list[dict[str, str | list[str]]] = []
    if ac_parsed:
        blocks.extend(ac_parsed.get("given_when_then", []))
    if description_parsed:
        blocks.extend(description_parsed.get("given_when_then", []))
    return blocks


def _extract_user_stories(issue: NormalizedIssue) -> list[dict[str, str]]:
    """Extract 'As a … I want … so that …' patterns."""
    stories: list[dict[str, str]] = []

    texts_to_search = [
        issue.summary,
        issue.description,
        issue.acceptance_criteria_raw or "",
    ]

    for text in texts_to_search:
        if not text:
            continue
        for m in _USER_STORY_RE.finditer(text):
            story: dict[str, str] = {
                "persona": m.group(1).strip(),
                "want": m.group(2).strip(),
            }
            if m.group(3):
                story["so_that"] = m.group(3).strip()
            stories.append(story)

    return stories


def _extract_by_heading(
    sections: dict[str, str],
    heading_aliases: set[str],
) -> list[str]:
    """Extract bullet-point items from sections matching heading aliases."""
    items: list[str] = []
    for heading, body in sections.items():
        if heading.lower().strip() in heading_aliases:
            items.extend(_lines_from_section(body))
    return _deduplicate(items)


def _is_ac_heading(heading: str) -> bool:
    """Fuzzy-match a heading string against known AC heading patterns."""
    normalised = heading.lower().strip()
    return any(alias in normalised for alias in _AC_HEADINGS)


def _lines_from_section(body: str) -> list[str]:
    """Split a section body into individual items.

    Handles bullet-prefixed lines, numbered lines, and plain text lines.
    """
    items: list[str] = []
    for line in body.split("\n"):
        cleaned = line.strip()
        # Remove wiki-markup bullet/number prefixes
        cleaned = re.sub(r"^[*#]+\s*", "", cleaned)
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            items.append(cleaned)
    return items


def _split_into_items(text: str) -> list[str]:
    """Split plain text into items by newlines, filtering blanks."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def _deduplicate(items: list[str]) -> list[str]:
    """Remove duplicate items while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalised = item.strip()
        if normalised and normalised not in seen:
            seen.add(normalised)
            result.append(normalised)
    return result
