"""Normalized Jira issue models.

These models represent the canonical intermediate form of a Jira issue
after raw API data has been parsed and cleaned, but before any AI-driven
knowledge extraction takes place. Every downstream pipeline stage
(structuring, validation, export) consumes ``NormalizedIssue`` as input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class IssueLink(BaseModel):
    """A directed link between two Jira issues.

    Attributes:
        link_type: The semantic category of the link
            (e.g. ``'Blocks'``, ``'Relates'``, ``'Cloners'``).
        direction: Whether *this* issue is the inward or outward end
            of the relationship.
        target_key: The Jira key of the issue on the other end
            (e.g. ``'PROJ-42'``).
    """

    link_type: str = Field(
        ...,
        description="Semantic category of the link (e.g. 'Blocks', 'Relates', 'Cloners', 'Continuation', 'Depends', 'Fixes').",
    )
    direction: Literal["inward", "outward"] = Field(
        ...,
        description="Whether this issue sits on the inward or outward side of the relationship.",
    )
    target_key: str = Field(
        ...,
        min_length=1,
        description="Jira key of the linked issue (e.g. 'PROJ-42').",
    )

    @field_validator("link_type")
    @classmethod
    def link_type_not_empty(cls, v: str) -> str:
        """Ensure ``link_type`` is a non-blank string."""
        if not v.strip():
            msg = "link_type must not be blank"
            raise ValueError(msg)
        return v.strip()


class Attachment(BaseModel):
    """Metadata for a file attached to a Jira issue.

    Attributes:
        filename: Original name of the uploaded file.
        url: Download URL (usually a Jira REST endpoint).
        uploaded_by: Display-name of the user who uploaded the file.
        uploaded_date: ISO-8601 timestamp of the upload.
    """

    filename: str = Field(..., min_length=1, description="Original file name.")
    url: str = Field(..., min_length=1, description="Download URL for the attachment.")
    uploaded_by: str | None = Field(
        default=None, description="Display-name of the uploader."
    )
    uploaded_date: str | None = Field(
        default=None,
        description="ISO-8601 upload timestamp.",
    )


class NormalizedIssue(BaseModel):
    """Canonical, provider-agnostic representation of a single Jira issue.

    This is the *single source of truth* consumed by every downstream
    stage of the contextualization pipeline.  Fields that may not be
    present for every issue type are typed as ``Optional`` / have empty
    defaults.

    Attributes:
        issue_key: Unique Jira key (e.g. ``'PROJ-123'``).
        issue_id: Internal numeric Jira ID.
        summary: One-line summary / title.
        issue_type: Type name (``'Story'``, ``'Bug'``, ``'Epic'``, …).
        status: Current workflow status (``'To Do'``, ``'In Progress'``, …).
        project_key: Short project identifier.
        project_name: Human-readable project name.
        priority: Priority label (``'Critical'``, ``'Major'``, …).
        resolution: Resolution name when issue is resolved.
        assignee: Display-name of the current assignee.
        reporter: Display-name of the reporter.
        creator: Display-name of the issue creator.
        created: ISO-8601 creation timestamp.
        updated: ISO-8601 last-updated timestamp.
        resolved: ISO-8601 resolution timestamp.
        components: List of component names the issue belongs to.
        labels: List of labels applied to the issue.
        description: Full plain-text description (wiki markup stripped).
        description_sections: Key/value map of named sections parsed from
            wiki-markup headings (e.g. ``{'Background': '…', 'Steps': '…'}``).
        epic_link: Key of the parent epic, if any.
        parent_link: Key of a generic parent issue (sub-task → parent).
        sprints: Sprint names the issue has been associated with.
        issue_links: Typed, directed links to other issues.
        attachments: Files attached to the issue.
        acceptance_criteria_raw: Raw acceptance-criteria text before parsing.
        environment: Deployment / environment notes.
        story_points: Estimated effort.
        custom_fields: Catch-all map for project-specific custom fields.
    """

    # ── identity ──────────────────────────────────────────────────────
    issue_key: str = Field(..., min_length=1, description="Jira issue key (e.g. 'PROJ-123').")
    issue_id: str = Field(..., min_length=1, description="Internal Jira numeric ID.")
    summary: str = Field(..., min_length=1, description="One-line issue summary.")
    issue_type: str = Field(..., min_length=1, description="Issue type name (Story, Bug, Epic, …).")
    status: str = Field(..., min_length=1, description="Current workflow status.")

    # ── project ───────────────────────────────────────────────────────
    project_key: str = Field(..., min_length=1, description="Short project identifier.")
    project_name: str = Field(..., min_length=1, description="Human-readable project name.")

    # ── triage fields ─────────────────────────────────────────────────
    priority: str = Field(default="Unset", description="Priority label.")
    resolution: str | None = Field(default=None, description="Resolution name, if resolved.")
    assignee: str | None = Field(default=None, description="Current assignee display-name.")
    reporter: str | None = Field(default=None, description="Reporter display-name.")
    creator: str | None = Field(default=None, description="Creator display-name.")

    # ── timestamps ────────────────────────────────────────────────────
    created: str = Field(..., description="ISO-8601 creation timestamp.")
    updated: str = Field(..., description="ISO-8601 last-updated timestamp.")
    resolved: str | None = Field(default=None, description="ISO-8601 resolution timestamp.")

    # ── categorisation ────────────────────────────────────────────────
    components: list[str] = Field(default_factory=list, description="Component names.")
    labels: list[str] = Field(default_factory=list, description="Labels applied to the issue.")

    # ── description ───────────────────────────────────────────────────
    description: str = Field(default="", description="Full plain-text description.")
    description_sections: dict[str, str] = Field(
        default_factory=dict,
        description="Named sections parsed from wiki-markup headings.",
    )

    # ── hierarchy & planning ──────────────────────────────────────────
    epic_link: str | None = Field(default=None, description="Parent epic key.")
    parent_link: str | None = Field(default=None, description="Generic parent issue key.")
    sprints: list[str] = Field(default_factory=list, description="Associated sprint names.")

    # ── relationships & attachments ───────────────────────────────────
    issue_links: list[IssueLink] = Field(default_factory=list, description="Typed links to other issues.")
    attachments: list[Attachment] = Field(default_factory=list, description="Attached files.")

    # ── extended fields ───────────────────────────────────────────────
    acceptance_criteria_raw: str | None = Field(
        default=None,
        description="Raw acceptance-criteria text before AI parsing.",
    )
    environment: str | None = Field(default=None, description="Environment / deployment notes.")
    story_points: float | None = Field(default=None, ge=0, description="Estimated story points.")
    custom_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Catch-all map for unmapped custom fields.",
    )

    # ── validators ────────────────────────────────────────────────────
    @field_validator("issue_key")
    @classmethod
    def issue_key_format(cls, v: str) -> str:
        """Enforce the ``PROJECT-123`` key format."""
        if "-" not in v:
            msg = f"issue_key must contain a hyphen (got '{v}')"
            raise ValueError(msg)
        return v.strip().upper()

    @field_validator("priority")
    @classmethod
    def normalise_priority(cls, v: str) -> str:
        """Strip whitespace and fall back to ``'Unset'`` for blanks."""
        cleaned = v.strip() if v else ""
        return cleaned if cleaned else "Unset"
