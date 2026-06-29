"""Structured issue knowledge models.

These models represent the *enriched*, AI-extracted knowledge that the
contextualization crew produces from a ``NormalizedIssue``.  They capture
business objectives, requirements, acceptance criteria in Given/When/Then
form, traceability links, and quality metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class AcceptanceCriterion(BaseModel):
    """A single, testable acceptance criterion.

    When the criterion can be expressed in Gherkin style the ``given``,
    ``when``, and ``then`` fields are populated; otherwise only
    ``description`` is required.

    Attributes:
        id: Short identifier (e.g. ``'AC-1'``, ``'AC-2'``).
        description: Human-readable statement of the criterion.
        given: Precondition clause (Gherkin *Given*).
        when: Action clause (Gherkin *When*).
        then: Expected outcome clause (Gherkin *Then*).
        is_testable: Whether the criterion is concrete enough to derive
            a test case from.
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Short identifier such as 'AC-1'.",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Plain-English statement of the criterion.",
    )
    given: str | None = Field(default=None, description="Gherkin *Given* clause.")
    when: str | None = Field(default=None, description="Gherkin *When* clause.")
    then: str | None = Field(default=None, description="Gherkin *Then* clause.")
    is_testable: bool = Field(
        default=True,
        description="True if the criterion is concrete enough to test.",
    )

    @field_validator("id")
    @classmethod
    def id_format(cls, v: str) -> str:
        """Normalise the ID to uppercase with a hyphen (e.g. ``AC-1``)."""
        return v.strip().upper()


class Dependency(BaseModel):
    """A dependency relationship between the current issue and another.

    Attributes:
        target_key: Jira key of the dependency target.
        dependency_type: Semantic type (``'blocks'``, ``'depends_on'``, …).
        direction: ``'inward'`` or ``'outward'``.
        description: Optional human-readable explanation of the dependency.
    """

    target_key: str = Field(
        ...,
        min_length=1,
        description="Jira key of the dependency target.",
    )
    dependency_type: str = Field(
        ...,
        min_length=1,
        description="Semantic dependency type (e.g. 'blocks', 'depends_on').",
    )
    direction: str = Field(
        ...,
        min_length=1,
        description="Relationship direction ('inward' or 'outward').",
    )
    description: str | None = Field(
        default=None,
        description="Free-text explanation of the dependency.",
    )


class TraceabilityLinks(BaseModel):
    """Cross-issue traceability graph for a single issue.

    Groups link targets by semantic role so downstream consumers
    (validators, exporters) can reason about the issue's position in
    the project hierarchy without re-parsing raw links.

    Attributes:
        epic_key: Parent epic key, if any.
        parent_key: Generic parent key (sub-task relationship).
        related_issues: Keys of issues with a *Relates* link.
        blocked_by: Keys of issues that block this one.
        blocks: Keys of issues this one blocks.
        depends_on: Keys of issues this one depends on.
        cloned_from: Keys of issues this one was cloned from.
    """

    epic_key: str | None = Field(default=None, description="Parent epic key.")
    parent_key: str | None = Field(default=None, description="Generic parent key.")
    related_issues: list[str] = Field(default_factory=list, description="Related issue keys.")
    blocked_by: list[str] = Field(default_factory=list, description="Issues blocking this one.")
    blocks: list[str] = Field(default_factory=list, description="Issues this one blocks.")
    depends_on: list[str] = Field(default_factory=list, description="Issues this depends on.")
    cloned_from: list[str] = Field(default_factory=list, description="Source clone keys.")


class Timeline(BaseModel):
    """Temporal metadata for an issue's lifecycle.

    Attributes:
        created: ISO-8601 creation timestamp.
        updated: ISO-8601 last-update timestamp.
        resolved: ISO-8601 resolution timestamp (``None`` if unresolved).
        sprints: Sprint names the issue has passed through.
        status_history: Ordered list of status transitions (oldest first).
    """

    created: str = Field(..., description="ISO-8601 creation timestamp.")
    updated: str = Field(..., description="ISO-8601 last-update timestamp.")
    resolved: str | None = Field(default=None, description="ISO-8601 resolution timestamp.")
    sprints: list[str] = Field(default_factory=list, description="Sprint names.")
    status_history: list[str] = Field(
        default_factory=list,
        description="Ordered status transitions (oldest → newest).",
    )


class StructuredIssueKnowledge(BaseModel):
    """Complete, AI-enriched knowledge extracted from a single Jira issue.

    This is the primary output of the contextualization pipeline.  It
    merges factual data (key, summary, timeline) with AI-inferred
    artefacts (business objective, requirements, acceptance criteria) and
    quality metadata (confidence / completeness scores).

    Attributes:
        issue_key: Jira issue key.
        summary: One-line summary.
        business_objective: AI-inferred business goal the issue serves.
        scope: Description of what is in and out of scope.
        functional_requirements: Extracted functional requirements.
        non_functional_requirements: Extracted NFRs.
        acceptance_criteria: Parsed and structured acceptance criteria.
        business_rules: Domain rules the implementation must respect.
        constraints: Technical or process constraints.
        risks_and_assumptions: Known risks and stated assumptions.
        dependencies: Typed dependency relationships.
        decisions: Recorded design / architecture decisions.
        open_questions: Unresolved questions requiring clarification.
        traceability_links: Cross-issue traceability graph.
        timeline: Temporal lifecycle data.
        confidence_score: 0.0 – 1.0 indicating extraction confidence.
        completeness_score: 0.0 – 1.0 indicating information completeness.
        missing_context: List of information gaps identified by the AI.
        extraction_notes: Free-form notes from the extraction process.
    """

    # ── identity ──────────────────────────────────────────────────────
    issue_key: str = Field(..., min_length=1, description="Jira issue key.")
    summary: str = Field(..., min_length=1, description="One-line issue summary.")

    # ── AI-inferred knowledge ─────────────────────────────────────────
    business_objective: str = Field(
        default="",
        description="AI-inferred business goal the issue serves.",
    )
    scope: str = Field(default="", description="In-scope and out-of-scope boundaries.")
    functional_requirements: list[str] = Field(
        default_factory=list,
        description="Extracted functional requirements.",
    )
    non_functional_requirements: list[str] = Field(
        default_factory=list,
        description="Extracted non-functional requirements.",
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        default_factory=list,
        description="Parsed acceptance criteria.",
    )
    business_rules: list[str] = Field(
        default_factory=list,
        description="Domain / business rules.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Technical or process constraints.",
    )
    risks_and_assumptions: list[str] = Field(
        default_factory=list,
        description="Known risks and stated assumptions.",
    )

    # ── relationships ─────────────────────────────────────────────────
    dependencies: list[Dependency] = Field(
        default_factory=list,
        description="Typed dependency relationships.",
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="Design / architecture decisions.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions.",
    )
    traceability_links: TraceabilityLinks = Field(
        default_factory=TraceabilityLinks,
        description="Cross-issue traceability graph.",
    )
    timeline: Timeline | None = Field(
        default=None,
        description="Temporal lifecycle data.",
    )

    # ── quality metadata ──────────────────────────────────────────────
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (0.0 – 1.0).",
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Information completeness (0.0 – 1.0).",
    )
    missing_context: list[str] = Field(
        default_factory=list,
        description="Information gaps identified during extraction.",
    )
    extraction_notes: list[str] = Field(
        default_factory=list,
        description="Free-form notes from the extraction process.",
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

    @model_validator(mode="after")
    def auto_compute_completeness(self) -> StructuredIssueKnowledge:
        """Recompute ``completeness_score`` if it was left at the default.

        The heuristic checks whether key knowledge fields are populated.
        A manually-set score (> 0) is preserved.
        """
        if self.completeness_score > 0.0:
            return self

        filled = 0
        total = 7  # number of key fields checked below
        if self.business_objective:
            filled += 1
        if self.scope:
            filled += 1
        if self.functional_requirements:
            filled += 1
        if self.acceptance_criteria:
            filled += 1
        if self.dependencies:
            filled += 1
        if self.business_rules:
            filled += 1
        if self.constraints:
            filled += 1

        self.completeness_score = round(filled / total, 2)
        return self
