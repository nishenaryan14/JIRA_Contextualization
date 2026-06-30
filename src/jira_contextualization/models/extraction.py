"""Pydantic models for the extraction pipeline stage.

These models represent the structured output of the per-issue extraction
crews (Task Groups 1 and 2) as well as the merged result that combines
both groups before consolidation.

Models
------
- :class:`ExtractedAcceptanceCriterion` — A single AC with optional BDD clauses.
- :class:`RequirementExtractionResult` — Task Group 1 output (core requirements).
- :class:`DiscussionExtractionResult` — Task Group 2 output (discussion & context).
- :class:`MergedExtractionResult` — Union of both groups for one issue.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedAcceptanceCriterion(BaseModel):
    """A single acceptance criterion extracted from a Jira ticket.

    Supports optional BDD-style Given/When/Then decomposition so that
    downstream testing crews can generate scenarios directly.
    """

    id: str = Field(description="Unique ID like AC-1, AC-2")
    description: str = Field(description="The acceptance criterion text")
    given: str | None = Field(default=None, description="Given clause (BDD)")
    when: str | None = Field(default=None, description="When clause (BDD)")
    then: str | None = Field(default=None, description="Then clause (BDD)")
    is_testable: bool = Field(
        default=True, description="Whether this AC is testable"
    )


class RequirementExtractionResult(BaseModel):
    """Output of Task Group 1: Core requirement extraction.

    Captures the business objective, scope boundaries, functional and
    non-functional requirements, acceptance criteria, business rules,
    and constraints for a single Jira issue.
    """

    issue_key: str = Field(description="The Jira issue key")
    business_objective: str = Field(
        description="Business goal this ticket serves"
    )
    scope: str = Field(default="", description="What is in/out of scope")
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    acceptance_criteria: list[ExtractedAcceptanceCriterion] = Field(
        default_factory=list
    )
    business_rules: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class DiscussionExtractionResult(BaseModel):
    """Output of Task Group 2: Discussion & context extraction.

    Captures risks, assumptions, decisions, open questions, and design
    notes surfaced from Jira comments and linked documents.
    """

    issue_key: str = Field(description="The Jira issue key")
    risks_and_assumptions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    design_notes: list[str] = Field(default_factory=list)


class MergedExtractionResult(BaseModel):
    """Merged result from all extraction task groups for one issue.

    Combines the outputs of :class:`RequirementExtractionResult` and
    :class:`DiscussionExtractionResult` into a single flat model that
    can be passed to the consolidation stage.
    """

    issue_key: str
    business_objective: str = ""
    scope: str = ""
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    acceptance_criteria: list[ExtractedAcceptanceCriterion] = Field(
        default_factory=list
    )
    business_rules: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks_and_assumptions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    design_notes: list[str] = Field(default_factory=list)
