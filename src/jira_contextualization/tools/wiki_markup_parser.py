"""Jira wiki-markup parser.

Deterministic utilities for extracting structured information from
Jira-flavoured wiki markup.  This module handles headings, bullet and
numbered lists, images, links, code blocks, bold/italic, and
Given/When/Then (Gherkin) patterns commonly found in acceptance criteria.

Public API
----------
- ``parse_jira_markup(text)`` → full structured parse result.
- ``extract_sections(text)`` → heading-keyed section map.
- ``clean_markup(text)``      → plain text with all markup removed.
"""

from __future__ import annotations

import re
from typing import Any

# ── heading regex ────────────────────────────────────────────────────────
# Jira wiki uses h1. … h6.  We anchor on the start of a line.
_HEADING_RE = re.compile(r"^h([1-6])\.\s*(.+)", re.MULTILINE)

# ── inline markup ────────────────────────────────────────────────────────
_BOLD_RE = re.compile(r"\*([^*\n]+)\*")
_ITALIC_RE = re.compile(r"_([^_\n]+)_")

# ── images & links ───────────────────────────────────────────────────────
_IMAGE_RE = re.compile(r"!([^!\s|]+)(?:\|[^!]*)?!")
_LINK_RE = re.compile(r"\[([^|\]]*)\|([^\]]+)\]")
_BARE_LINK_RE = re.compile(r"\[(https?://[^\]]+)\]")

# ── code / noformat blocks ──────────────────────────────────────────────
_CODE_BLOCK_RE = re.compile(r"\{code(?::[^}]*)?\}(.*?)\{code\}", re.DOTALL)
_NOFORMAT_RE = re.compile(r"\{noformat\}(.*?)\{noformat\}", re.DOTALL)

# ── list items ───────────────────────────────────────────────────────────
_BULLET_RE = re.compile(r"^\*+\s+(.+)", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^#+\s+(.+)", re.MULTILINE)

# ── GWT patterns ─────────────────────────────────────────────────────────
_GWT_GIVEN_RE = re.compile(r"(?:^|\n)\s*\*?Given\*?\s+(.+)", re.IGNORECASE)
_GWT_WHEN_RE = re.compile(r"(?:^|\n)\s*\*?When\*?\s+(.+)", re.IGNORECASE)
_GWT_THEN_RE = re.compile(r"(?:^|\n)\s*\*?Then\*?\s+(.+)", re.IGNORECASE)
_GWT_AND_RE = re.compile(r"(?:^|\n)\s*\*?And\*?\s+(.+)", re.IGNORECASE)

# ── table rows ───────────────────────────────────────────────────────────
_TABLE_ROW_RE = re.compile(r"^\|(?:\|?[^|\n]+)+\|?\s*$", re.MULTILINE)

# ── color / panel / misc macros ──────────────────────────────────────────
_COLOR_RE = re.compile(r"\{color(?::[^}]*)?\}(.*?)\{color\}", re.DOTALL)
_PANEL_RE = re.compile(r"\{panel(?::[^}]*)?\}(.*?)\{panel\}", re.DOTALL)
_QUOTE_RE = re.compile(r"\{quote\}(.*?)\{quote\}", re.DOTALL)


def parse_jira_markup(text: str) -> dict[str, Any]:
    """Parse Jira wiki markup into a structured dictionary.

    Args:
        text: Raw Jira wiki-markup string.

    Returns:
        Dictionary with keys:
        - ``sections``        – ``dict[str, str]`` of heading → body text.
        - ``bullet_points``   – ``list[str]`` of bullet-list items.
        - ``numbered_items``  – ``list[str]`` of numbered-list items.
        - ``given_when_then`` – ``list[dict]`` of structured GWT blocks.
        - ``images``          – ``list[str]`` of image filenames/URLs.
        - ``links``           – ``list[str]`` of hyperlink URLs.
        - ``clean_text``      – ``str`` with all markup stripped.
    """
    if not text or not text.strip():
        return {
            "sections": {},
            "bullet_points": [],
            "numbered_items": [],
            "given_when_then": [],
            "images": [],
            "links": [],
            "clean_text": "",
        }

    sections = extract_sections(text)
    bullet_points = _extract_bullet_points(text)
    numbered_items = _extract_numbered_items(text)
    given_when_then = _extract_gwt_blocks(text)
    images = _extract_images(text)
    links = _extract_links(text)
    cleaned = clean_markup(text)

    return {
        "sections": sections,
        "bullet_points": bullet_points,
        "numbered_items": numbered_items,
        "given_when_then": given_when_then,
        "images": images,
        "links": links,
        "clean_text": cleaned,
    }


def extract_sections(text: str) -> dict[str, str]:
    """Split wiki-markup text by h1.–h6. headings into named sections.

    Headings are normalised to title-case keys.  Text before the first
    heading is stored under the key ``'_preamble'``.

    Args:
        text: Raw wiki-markup string.

    Returns:
        Ordered mapping of heading title → section body text.
    """
    if not text or not text.strip():
        return {}

    sections: dict[str, str] = {}
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        # No headings — return everything as preamble
        stripped = text.strip()
        if stripped:
            sections["_preamble"] = stripped
        return sections

    # Text before the first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["_preamble"] = preamble

    for idx, match in enumerate(matches):
        heading_title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections[heading_title] = body

    return sections


def clean_markup(text: str) -> str:
    """Strip all Jira wiki markup and return plain text.

    Removes headings markers, bold/italic markers, images, links (keeping
    the display text), code/noformat blocks (keeping contents), color/panel
    macros, and table-pipe characters.

    Args:
        text: Raw wiki-markup string.

    Returns:
        Plain text with markup artefacts removed.
    """
    if not text:
        return ""

    result = text

    # Remove code/noformat blocks — keep inner text
    result = _CODE_BLOCK_RE.sub(r"\1", result)
    result = _NOFORMAT_RE.sub(r"\1", result)

    # Remove color/panel/quote macros — keep inner text
    result = _COLOR_RE.sub(r"\1", result)
    result = _PANEL_RE.sub(r"\1", result)
    result = _QUOTE_RE.sub(r"\1", result)

    # Remove heading markers  (h1. Title  →  Title)
    result = _HEADING_RE.sub(r"\2", result)

    # Replace links with display text  [text|url] → text
    result = _LINK_RE.sub(r"\1", result)
    # Bare links  [http://...] → url
    result = _BARE_LINK_RE.sub(r"\1", result)

    # Remove image references
    result = _IMAGE_RE.sub("", result)

    # Strip bold/italic markers
    result = _BOLD_RE.sub(r"\1", result)
    result = _ITALIC_RE.sub(r"\1", result)

    # Remove bullet / numbered list markers
    result = re.sub(r"^\*+\s+", "", result, flags=re.MULTILINE)
    result = re.sub(r"^#+\s+", "", result, flags=re.MULTILINE)

    # Remove table pipe characters
    result = re.sub(r"\|{1,2}", " ", result)

    # Remove horizontal rules (----)
    result = re.sub(r"^-{4,}\s*$", "", result, flags=re.MULTILINE)

    # Collapse excessive whitespace
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


# ── private helpers ──────────────────────────────────────────────────────


def _extract_bullet_points(text: str) -> list[str]:
    """Extract all bullet-list items (``* item``)."""
    return [m.group(1).strip() for m in _BULLET_RE.finditer(text)]


def _extract_numbered_items(text: str) -> list[str]:
    """Extract all numbered-list items (``# item``)."""
    return [m.group(1).strip() for m in _NUMBERED_RE.finditer(text)]


def _extract_images(text: str) -> list[str]:
    """Extract image filenames/URLs from ``!image.png!`` references."""
    return [m.group(1).strip() for m in _IMAGE_RE.finditer(text)]


def _extract_links(text: str) -> list[str]:
    """Extract URLs from ``[text|url]`` and bare ``[url]`` references."""
    urls: list[str] = []
    for m in _LINK_RE.finditer(text):
        urls.append(m.group(2).strip())
    for m in _BARE_LINK_RE.finditer(text):
        urls.append(m.group(1).strip())
    return urls


def _extract_gwt_blocks(text: str) -> list[dict[str, str | list[str]]]:
    """Extract Given/When/Then blocks from text.

    Handles both explicit Gherkin syntax and bullet-prefixed variants
    commonly found in Jira acceptance criteria.  Multiple GWT blocks
    in the same text are returned as separate dictionaries.

    Returns:
        List of dicts, each with keys ``given``, ``when``, ``then``,
        and optionally ``and_clauses``.
    """
    blocks: list[dict[str, str | list[str]]] = []

    # Split text into lines for stateful parsing
    lines = text.split("\n")
    current_block: dict[str, str | list[str]] | None = None
    current_key: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        # Remove leading bullet/number markers for matching
        cleaned = re.sub(r"^[*#]+\s*", "", line)

        given_match = re.match(r"\*?Given\*?\s+(.+)", cleaned, re.IGNORECASE)
        when_match = re.match(r"\*?When\*?\s+(.+)", cleaned, re.IGNORECASE)
        then_match = re.match(r"\*?Then\*?\s+(.+)", cleaned, re.IGNORECASE)
        and_match = re.match(r"\*?And\*?\s+(.+)", cleaned, re.IGNORECASE)

        if given_match:
            # Start a new GWT block
            if current_block and ("given" in current_block or "when" in current_block):
                blocks.append(current_block)
            current_block = {
                "given": given_match.group(1).strip(),
                "and_clauses": [],
            }
            current_key = "given"

        elif when_match and current_block is not None:
            current_block["when"] = when_match.group(1).strip()
            current_key = "when"

        elif then_match and current_block is not None:
            current_block["then"] = then_match.group(1).strip()
            current_key = "then"

        elif and_match and current_block is not None and current_key is not None:
            and_clauses = current_block.setdefault("and_clauses", [])
            assert isinstance(and_clauses, list)  # noqa: S101
            and_clauses.append(and_match.group(1).strip())

    # Flush the last block
    if current_block and ("given" in current_block or "when" in current_block):
        blocks.append(current_block)

    return blocks
