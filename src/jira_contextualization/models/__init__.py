"""Pydantic v2 data models for the Jira Contextualization pipeline.

Modules
-------
- :mod:`normalized_issue`    — Canonical intermediate issue representation.
- :mod:`structured_knowledge`— AI-enriched per-issue knowledge.
- :mod:`project_knowledge`   — Project-wide aggregations and roll-ups.
- :mod:`validation_report`   — Quality findings and reports.
"""

from jira_contextualization.models.normalized_issue import (
    Attachment,
    IssueLink,
    NormalizedIssue,
)
from jira_contextualization.models.project_knowledge import (
    ComponentSummary,
    EpicSummary,
    ProjectKnowledge,
    QualityMetrics,
)
from jira_contextualization.models.structured_knowledge import (
    AcceptanceCriterion,
    Dependency,
    StructuredIssueKnowledge,
    Timeline,
    TraceabilityLinks,
)
from jira_contextualization.models.validation_report import (
    IssueValidationResult,
    QualityReport,
    ValidationIssue,
)

__all__ = [
    # normalized_issue
    "Attachment",
    "IssueLink",
    "NormalizedIssue",
    # structured_knowledge
    "AcceptanceCriterion",
    "Dependency",
    "StructuredIssueKnowledge",
    "Timeline",
    "TraceabilityLinks",
    # project_knowledge
    "ComponentSummary",
    "EpicSummary",
    "ProjectKnowledge",
    "QualityMetrics",
    # validation_report
    "IssueValidationResult",
    "QualityReport",
    "ValidationIssue",
]
