"""Validation and quality-report models.

These models capture the output of the validation pipeline: per-issue
quality checks (missing acceptance criteria, ambiguity, orphan detection,
etc.) rolled up into a project-wide quality report with severity counts
and an overall quality score.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ValidationIssue(BaseModel):
    """A single quality finding for one Jira issue.

    Attributes:
        issue_key: The Jira key the finding relates to.
        severity: Impact level of the finding.
        category: Machine-readable category tag used for filtering and
            aggregation (e.g. ``'missing_ac'``, ``'ambiguity'``).
        message: Human-readable explanation of the finding.
        suggestion: Optional remediation guidance.
    """

    issue_key: str = Field(..., min_length=1, description="Jira key this finding relates to.")
    severity: Literal["critical", "warning", "info"] = Field(
        ...,
        description="Impact level: 'critical', 'warning', or 'info'.",
    )
    category: str = Field(
        ...,
        min_length=1,
        description=(
            "Machine-readable category tag "
            "(e.g. 'missing_ac', 'missing_priority', 'ambiguity', "
            "'duplicate', 'orphan')."
        ),
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of the finding.",
    )
    suggestion: str | None = Field(
        default=None,
        description="Optional remediation guidance.",
    )

    @field_validator("category")
    @classmethod
    def normalise_category(cls, v: str) -> str:
        """Lower-case and strip the category for consistent aggregation."""
        return v.strip().lower().replace(" ", "_")


class IssueValidationResult(BaseModel):
    """Aggregated validation outcome for a single Jira issue.

    Attributes:
        issue_key: The Jira key that was validated.
        completeness_score: 0.0 – 1.0 completeness rating.
        confidence_score: 0.0 – 1.0 confidence rating.
        issues_found: Individual findings for this issue.
        is_valid: ``True`` when no *critical* findings exist.
    """

    issue_key: str = Field(..., min_length=1, description="Validated issue key.")
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Completeness rating (0.0 – 1.0).",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence rating (0.0 – 1.0).",
    )
    issues_found: list[ValidationIssue] = Field(
        default_factory=list,
        description="Validation findings for this issue.",
    )
    is_valid: bool = Field(
        default=True,
        description="True when no critical findings exist.",
    )

    @model_validator(mode="after")
    def derive_validity(self) -> IssueValidationResult:
        """Auto-set ``is_valid`` to ``False`` if any critical finding exists."""
        has_critical = any(
            issue.severity == "critical" for issue in self.issues_found
        )
        if has_critical:
            self.is_valid = False
        return self


class QualityReport(BaseModel):
    """Project-wide quality report produced by the validation pipeline.

    Attributes:
        generated_at: ISO-8601 timestamp of report generation.
        total_issues_analyzed: Number of issues that were validated.
        total_validation_issues: Total findings across all issues.
        critical_count: Number of critical-severity findings.
        warning_count: Number of warning-severity findings.
        info_count: Number of info-severity findings.
        overall_quality_score: 0.0 – 1.0 aggregate quality score.
        issue_results: Per-issue validation results.
        summary: Human-readable executive summary.
    """

    generated_at: str = Field(..., description="ISO-8601 generation timestamp.")
    total_issues_analyzed: int = Field(
        default=0, ge=0, description="Issues validated."
    )
    total_validation_issues: int = Field(
        default=0, ge=0, description="Total findings."
    )
    critical_count: int = Field(default=0, ge=0, description="Critical findings.")
    warning_count: int = Field(default=0, ge=0, description="Warning findings.")
    info_count: int = Field(default=0, ge=0, description="Info findings.")
    overall_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate quality score (0.0 – 1.0).",
    )
    issue_results: list[IssueValidationResult] = Field(
        default_factory=list,
        description="Per-issue validation outcomes.",
    )
    summary: str = Field(
        default="",
        description="Human-readable executive summary.",
    )

    @model_validator(mode="after")
    def compute_counts(self) -> QualityReport:
        """Derive severity counts and totals from ``issue_results``.

        Only runs when the explicit counts are left at their defaults
        (all zero) so that manually-provided values are preserved.
        """
        if not self.issue_results:
            return self

        all_findings: list[ValidationIssue] = []
        for result in self.issue_results:
            all_findings.extend(result.issues_found)

        # Only auto-compute when caller left counts at zero
        counts_untouched = (
            self.critical_count == 0
            and self.warning_count == 0
            and self.info_count == 0
            and self.total_validation_issues == 0
        )

        if counts_untouched and all_findings:
            self.critical_count = sum(
                1 for f in all_findings if f.severity == "critical"
            )
            self.warning_count = sum(
                1 for f in all_findings if f.severity == "warning"
            )
            self.info_count = sum(
                1 for f in all_findings if f.severity == "info"
            )
            self.total_validation_issues = len(all_findings)

        if self.total_issues_analyzed == 0:
            self.total_issues_analyzed = len(self.issue_results)

        return self
