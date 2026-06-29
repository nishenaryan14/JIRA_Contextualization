"""Project-level knowledge aggregation models.

These models roll up individual ``StructuredIssueKnowledge`` records into
project-wide summaries: epic / component breakdowns, quality metrics, and
the full dependency graph.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from jira_contextualization.models.structured_knowledge import (
    StructuredIssueKnowledge,
)


class EpicSummary(BaseModel):
    """Aggregate summary for a single epic.

    Attributes:
        epic_key: Jira key of the epic.
        title: Human-readable epic title (may be ``None`` if not fetched).
        story_count: Number of stories / child issues linked to this epic.
        stories: Keys of child stories.
        components: Unique component names across child stories.
        avg_confidence: Mean confidence score of child issues.
    """

    epic_key: str = Field(..., min_length=1, description="Epic Jira key.")
    title: str | None = Field(default=None, description="Epic title.")
    story_count: int = Field(default=0, ge=0, description="Child story count.")
    stories: list[str] = Field(default_factory=list, description="Child story keys.")
    components: list[str] = Field(
        default_factory=list,
        description="Unique components across child stories.",
    )
    avg_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean confidence score of child issues.",
    )

    @model_validator(mode="after")
    def sync_story_count(self) -> EpicSummary:
        """Keep ``story_count`` in sync with the ``stories`` list."""
        if self.stories and self.story_count == 0:
            self.story_count = len(self.stories)
        return self


class ComponentSummary(BaseModel):
    """Aggregate summary for a single project component.

    Attributes:
        name: Component name as it appears in Jira.
        story_count: Number of issues tagged with this component.
        stories: Keys of issues tagged with this component.
        epics: Unique epic keys whose children touch this component.
    """

    name: str = Field(..., min_length=1, description="Component name.")
    story_count: int = Field(default=0, ge=0, description="Issue count.")
    stories: list[str] = Field(default_factory=list, description="Issue keys.")
    epics: list[str] = Field(
        default_factory=list,
        description="Epic keys whose children touch this component.",
    )

    @model_validator(mode="after")
    def sync_story_count(self) -> ComponentSummary:
        """Keep ``story_count`` in sync with the ``stories`` list."""
        if self.stories and self.story_count == 0:
            self.story_count = len(self.stories)
        return self


class QualityMetrics(BaseModel):
    """Project-wide quality statistics derived from all analysed issues.

    Attributes:
        total_issues: Total number of issues analysed.
        avg_confidence_score: Mean confidence score.
        avg_completeness_score: Mean completeness score.
        issues_with_ac: Count of issues with ≥ 1 acceptance criterion.
        issues_missing_priority: Count of issues whose priority is unset.
        issues_with_links: Count of issues with ≥ 1 traceability link.
        issues_missing_description: Count of issues with empty descriptions.
        quality_grade: Letter grade (``A`` – ``F``) derived from scores.
    """

    total_issues: int = Field(default=0, ge=0, description="Total issues analysed.")
    avg_confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean confidence."
    )
    avg_completeness_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean completeness."
    )
    issues_with_ac: int = Field(
        default=0, ge=0, description="Issues with ≥ 1 acceptance criterion."
    )
    issues_missing_priority: int = Field(
        default=0, ge=0, description="Issues with unset priority."
    )
    issues_with_links: int = Field(
        default=0, ge=0, description="Issues with ≥ 1 traceability link."
    )
    issues_missing_description: int = Field(
        default=0, ge=0, description="Issues with empty descriptions."
    )
    quality_grade: str = Field(
        default="F",
        description="Letter grade (A–F) derived from aggregate scores.",
    )

    @field_validator("quality_grade")
    @classmethod
    def valid_grade(cls, v: str) -> str:
        """Ensure the grade is a recognised letter."""
        allowed = {"A", "B", "C", "D", "F"}
        normalised = v.strip().upper()
        if normalised not in allowed:
            msg = f"quality_grade must be one of {allowed} (got '{v}')"
            raise ValueError(msg)
        return normalised


class ProjectKnowledge(BaseModel):
    """Complete knowledge base for a Jira project.

    This is the top-level artefact produced by the full contextualization
    pipeline.  It bundles per-issue knowledge, epic / component roll-ups,
    aggregate quality metrics, and the project-wide dependency graph.

    Attributes:
        project_key: Short project identifier.
        project_name: Human-readable project name.
        generated_at: ISO-8601 timestamp of generation.
        total_issues: Total issues included.
        issues: Full list of structured issue knowledge records.
        epics: Epic-level roll-up summaries.
        components: Component-level roll-up summaries.
        quality_metrics: Aggregate quality statistics.
        dependency_graph: Adjacency-list representation of the project
            dependency graph (key → list of keys it depends on / blocks).
        metadata: Free-form metadata (pipeline version, model used, …).
    """

    project_key: str = Field(..., min_length=1, description="Project identifier.")
    project_name: str = Field(..., min_length=1, description="Project name.")
    generated_at: str = Field(..., description="ISO-8601 generation timestamp.")
    total_issues: int = Field(default=0, ge=0, description="Total issues included.")
    issues: list[StructuredIssueKnowledge] = Field(
        default_factory=list,
        description="Per-issue structured knowledge.",
    )
    epics: list[EpicSummary] = Field(
        default_factory=list,
        description="Epic roll-up summaries.",
    )
    components: list[ComponentSummary] = Field(
        default_factory=list,
        description="Component roll-up summaries.",
    )
    quality_metrics: QualityMetrics = Field(
        default_factory=QualityMetrics,
        description="Aggregate quality statistics.",
    )
    dependency_graph: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Adjacency list: issue key → list of related keys.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Free-form pipeline metadata.",
    )

    @model_validator(mode="after")
    def sync_total_issues(self) -> ProjectKnowledge:
        """Keep ``total_issues`` consistent with ``len(issues)``."""
        if self.issues and self.total_issues == 0:
            self.total_issues = len(self.issues)
        return self
