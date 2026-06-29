"""HP Jira CSV parser.

Deterministic parser for the wide-format Jira CSV export used by the HP
project.  The export contains **1 646 columns** with many duplicate header
names (e.g. ``Component/s`` ×3, ``Labels`` ×7, ``Sprint`` ×8).  This
module resolves ambiguity by mapping columns *by index* rather than by
name, and collects multi-value fields across their repeated columns.

Public API
----------
- ``parse_jira_csv(csv_path)`` → ``list[NormalizedIssue]``
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any

from jira_contextualization.models.normalized_issue import (
    Attachment,
    IssueLink,
    NormalizedIssue,
)
from jira_contextualization.tools.wiki_markup_parser import (
    clean_markup,
    extract_sections,
)

logger = logging.getLogger(__name__)

# Increase the CSV field-size limit for the massive HP export.
# sys.maxsize overflows on Windows; use a safe 10 MB limit instead.
csv.field_size_limit(min(sys.maxsize, 10_000_000))

# ── Known column indices (0-based) ───────────────────────────────────────
# These were established from the March 2026 HP-Jira CSV header analysis.
_COL = {
    "summary": 0,
    "issue_key": 1,
    "issue_id": 2,
    "issue_type": 3,
    "status": 4,
    "project_key": 5,
    "project_name": 6,
    "priority": 11,
    "resolution": 12,
    "assignee": 13,
    "reporter": 14,
    "creator": 15,
    "created": 16,
    "updated": 17,
    "resolved": 19,
    "description": 33,
    "environment": 34,
    "acceptance_criteria": 198,
    "epic_link": 591,
    "parent_link": 1003,
    "parent_link_initiative": 1004,
}

# Multi-value column ranges (inclusive start, exclusive end)
_COMPONENT_COLS = range(22, 25)          # 22, 23, 24
_LABEL_COLS = range(26, 33)              # 26 – 32
_SPRINT_COLS = range(35, 43)             # 8 columns, approximate — we auto-detect below
_ISSUE_LINK_COLS = range(58, 168)        # 58 – 167
_ATTACHMENT_COLS = range(168, 182)       # 168 – 181  (14 columns)


def parse_jira_csv(csv_path: str) -> list[NormalizedIssue]:
    """Parse an HP Jira CSV export into a list of ``NormalizedIssue`` objects.

    The parser is resilient to:
    - Duplicate header names (uses positional indexing).
    - UTF-8 text with emoji characters (flags, check-marks, etc.).
    - Empty or whitespace-only cells.
    - Rows with fewer columns than expected (short rows are padded).

    Args:
        csv_path: Filesystem path to the ``.csv`` file.

    Returns:
        A list of ``NormalizedIssue`` instances, one per data row.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If the file has no data rows.
    """
    path = Path(csv_path)
    if not path.is_file():
        msg = f"CSV file not found: {csv_path}"
        raise FileNotFoundError(msg)

    issues: list[NormalizedIssue] = []

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        num_cols = len(headers)
        logger.info("CSV has %d columns", num_cols)

        # Build dynamic column maps for headers with duplicate names
        sprint_indices = _find_column_indices(headers, "Sprint")
        link_header_map = _build_link_header_map(headers)

        for row_num, row in enumerate(reader, start=2):
            # Pad short rows so index access never raises
            if len(row) < num_cols:
                row.extend([""] * (num_cols - len(row)))

            try:
                issue = _row_to_normalized_issue(
                    row,
                    sprint_indices=sprint_indices,
                    link_header_map=link_header_map,
                )
                issues.append(issue)
            except Exception:
                logger.exception("Failed to parse row %d (key=%s)", row_num, _safe_get(row, 1))
                continue

    if not issues:
        msg = f"No valid issues parsed from {csv_path}"
        raise ValueError(msg)

    logger.info("Successfully parsed %d issues from %s", len(issues), csv_path)
    return issues


# ── Row-level parsing ────────────────────────────────────────────────────


def _row_to_normalized_issue(
    row: list[str],
    *,
    sprint_indices: list[int],
    link_header_map: dict[int, tuple[str, str]],
) -> NormalizedIssue:
    """Convert a single CSV row into a ``NormalizedIssue``."""
    raw_description = _safe_get(row, _COL["description"])
    description_clean = clean_markup(raw_description) if raw_description else ""
    description_sects = extract_sections(raw_description) if raw_description else {}

    # Multi-value fields: collect non-empty values across repeated columns
    components = _collect_multi(row, _COMPONENT_COLS)
    labels = _collect_multi(row, _LABEL_COLS)
    sprints = _collect_multi(row, sprint_indices)

    # Issue links
    issue_links = _parse_issue_links(row, link_header_map)

    # Attachments
    attachments = _parse_attachments(row)

    # Epic / parent link: prefer the explicit Epic Link, fall back to Parent
    epic_link = _safe_get(row, _COL["epic_link"]) or None
    parent_link = (
        _safe_get(row, _COL["parent_link"])
        or _safe_get(row, _COL["parent_link_initiative"])
        or None
    )

    # Acceptance criteria from the dedicated custom field
    ac_raw = _safe_get(row, _COL["acceptance_criteria"]) or None

    return NormalizedIssue(
        issue_key=_safe_get(row, _COL["issue_key"]),
        issue_id=_safe_get(row, _COL["issue_id"]),
        summary=_safe_get(row, _COL["summary"]),
        issue_type=_safe_get(row, _COL["issue_type"]),
        status=_safe_get(row, _COL["status"]),
        project_key=_safe_get(row, _COL["project_key"]),
        project_name=_safe_get(row, _COL["project_name"]),
        priority=_safe_get(row, _COL["priority"]) or "Unset",
        resolution=_safe_get(row, _COL["resolution"]) or None,
        assignee=_safe_get(row, _COL["assignee"]) or None,
        reporter=_safe_get(row, _COL["reporter"]) or None,
        creator=_safe_get(row, _COL["creator"]) or None,
        created=_safe_get(row, _COL["created"]),
        updated=_safe_get(row, _COL["updated"]),
        resolved=_safe_get(row, _COL["resolved"]) or None,
        components=components,
        labels=labels,
        description=description_clean,
        description_sections=description_sects,
        epic_link=epic_link,
        parent_link=parent_link,
        sprints=sprints,
        issue_links=issue_links,
        attachments=attachments,
        acceptance_criteria_raw=ac_raw,
        environment=_safe_get(row, _COL["environment"]) or None,
    )


# ── Multi-value & link helpers ───────────────────────────────────────────


def _collect_multi(row: list[str], indices: range | list[int]) -> list[str]:
    """Collect non-empty, unique values from *indices* in *row*."""
    seen: set[str] = set()
    result: list[str] = []
    for idx in indices:
        val = _safe_get(row, idx).strip()
        if val and val not in seen:
            seen.add(val)
            result.append(val)
    return result


def _find_column_indices(headers: list[str], name: str) -> list[int]:
    """Return every index where ``headers[i] == name``."""
    return [i for i, h in enumerate(headers) if h.strip() == name]


def _build_link_header_map(headers: list[str]) -> dict[int, tuple[str, str]]:
    """Build a map of column-index → ``(link_type, direction)`` for issue-link columns.

    Issue-link headers follow the pattern:
        ``"Outward issue link (Blocks)"``
        ``"Inward issue link (Depends)"``

    Returns:
        ``{col_index: (link_type, direction)}`` for every link column.
    """
    import re as _re

    link_pattern = _re.compile(
        r"(Inward|Outward)\s+issue\s+link\s*\(([^)]+)\)",
        _re.IGNORECASE,
    )

    result: dict[int, tuple[str, str]] = {}
    for idx in _ISSUE_LINK_COLS:
        if idx >= len(headers):
            break
        header = headers[idx].strip()
        m = link_pattern.search(header)
        if m:
            direction = m.group(1).lower()  # "inward" or "outward"
            link_type = m.group(2).strip()   # e.g. "Blocks", "Depends"
            result[idx] = (link_type, direction)
    return result


def _parse_issue_links(
    row: list[str],
    link_header_map: dict[int, tuple[str, str]],
) -> list[IssueLink]:
    """Parse issue-link columns into ``IssueLink`` objects.

    Each link column may contain a single Jira issue key (e.g. ``PROJ-42``)
    or be empty.  We skip empty cells.
    """
    links: list[IssueLink] = []
    for col_idx, (link_type, direction) in link_header_map.items():
        target_key = _safe_get(row, col_idx).strip()
        if not target_key:
            continue
        # Some cells may contain multiple keys separated by ", "
        for key in target_key.split(","):
            key = key.strip()
            if key and "-" in key:
                links.append(
                    IssueLink(
                        link_type=link_type,
                        direction=direction,  # type: ignore[arg-type]
                        target_key=key,
                    )
                )
    return links


def _parse_attachments(row: list[str]) -> list[Attachment]:
    """Parse attachment columns.

    Each attachment column contains a semicolon-separated string::

        ``date;user;filename;url``

    Multiple attachments in a single cell are newline-separated.
    """
    attachments: list[Attachment] = []
    for idx in _ATTACHMENT_COLS:
        cell = _safe_get(row, idx).strip()
        if not cell:
            continue
        # Handle multiple attachments per cell (newline-separated)
        for entry in cell.split("\n"):
            entry = entry.strip()
            if not entry:
                continue
            attachment = _parse_single_attachment(entry)
            if attachment is not None:
                attachments.append(attachment)
    return attachments


def _parse_single_attachment(entry: str) -> Attachment | None:
    """Parse one ``date;user;filename;url`` string into an ``Attachment``.

    Gracefully handles missing parts: at minimum the filename and URL
    must be present.
    """
    parts = [p.strip() for p in entry.split(";")]

    if len(parts) >= 4:
        return Attachment(
            uploaded_date=parts[0] or None,
            uploaded_by=parts[1] or None,
            filename=parts[2] or "unknown",
            url=parts[3] or "",
        )
    if len(parts) >= 2:
        # Best-effort: assume last part is URL, second-to-last is filename
        return Attachment(
            filename=parts[-2] or "unknown",
            url=parts[-1] or "",
        )
    # Unparseable
    logger.warning("Unparseable attachment entry: %s", entry)
    return None


# ── Utility ──────────────────────────────────────────────────────────────


def _safe_get(row: list[str], idx: int) -> str:
    """Return ``row[idx]`` or ``""`` if the index is out of range."""
    try:
        return row[idx] if idx < len(row) else ""
    except (IndexError, TypeError):
        return ""
