"""Knowledge validator.

Deterministic quality checks and completeness scoring for
``StructuredIssueKnowledge`` records.  Produces per-issue
``IssueValidationResult`` objects and project-wide ``QualityReport``
summaries.

Severity levels
---------------
- **critical** — information gap likely to cause downstream failures
  (e.g. missing acceptance criteria, empty description).
- **warning** — notable omission that reduces confidence
  (e.g. no functional requirements, unset priority).
- **info** — minor observation, non-blocking
  (e.g. orphan story, missing components).

Public API
----------
- ``validate_issue(knowledge)``
- ``calculate_completeness_score(knowledge)``
- ``generate_quality_report(results)``
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jira_contextualization.models.structured_knowledge import (
    StructuredIssueKnowledge,
)
from jira_contextualization.models.validation_report import (
    IssueValidationResult,
    QualityReport,
    ValidationIssue,
)


# ── Per-issue validation ─────────────────────────────────────────────────


def validate_issue(knowledge: StructuredIssueKnowledge) -> IssueValidationResult:
    """Run all deterministic quality checks on a single issue.

    Args:
        knowledge: AI-enriched knowledge record for one Jira issue.

    Returns:
        An ``IssueValidationResult`` containing every finding.
    """
    findings: list[ValidationIssue] = []
    key = knowledge.issue_key

    # ── Critical checks ──────────────────────────────────────────────

    # 1. Missing acceptance criteria
    if not knowledge.acceptance_criteria:
        findings.append(
            ValidationIssue(
                issue_key=key,
                severity="critical",
                category="missing_ac",
                message="No acceptance criteria found.",
                suggestion=(
                    "Add at least one testable acceptance criterion in "
                    "Given/When/Then or declarative form."
                ),
            )
        )

    # 2. Empty description / summary serving as sole content
    if not knowledge.business_objective and not knowledge.scope:
        # Check if there's *any* meaningful content beyond the summary
        has_content = bool(
            knowledge.functional_requirements
            or knowledge.non_functional_requirements
            or knowledge.business_rules
            or knowledge.constraints
        )
        if not has_content:
            findings.append(
                ValidationIssue(
                    issue_key=key,
                    severity="critical",
                    category="empty_description",
                    message="Issue has no meaningful description or extracted content.",
                    suggestion="Provide a detailed description with scope and requirements.",
                )
            )

    # ── Warning checks ───────────────────────────────────────────────

    # 3. Missing / unset priority
    # Note: the AI model doesn't carry priority directly, but we can
    #       infer it from extraction_notes or missing_context.  For now
    #       we rely on the completeness_score heuristic.
    if knowledge.completeness_score < 0.3:
        findings.append(
            ValidationIssue(
                issue_key=key,
                severity="warning",
                category="low_completeness",
                message=f"Completeness score is very low ({knowledge.completeness_score:.0%}).",
                suggestion="Review the issue for missing information and enrich the description.",
            )
        )

    # 4. No functional requirements extracted
    if not knowledge.functional_requirements:
        findings.append(
            ValidationIssue(
                issue_key=key,
                severity="warning",
                category="missing_requirements",
                message="No functional requirements were extracted.",
                suggestion="Add explicit functional requirements to the description.",
            )
        )

    # 5. No business objective
    if not knowledge.business_objective:
        findings.append(
            ValidationIssue(
                issue_key=key,
                severity="warning",
                category="missing_objective",
                message="No business objective was identified.",
                suggestion=(
                    "State the business goal this issue serves "
                    "(e.g. 'Improve onboarding conversion by 15%')."
                ),
            )
        )

    # ── Info checks ──────────────────────────────────────────────────

    # 6. Orphan story — no epic link, no traceability links
    is_orphan = (
        not knowledge.traceability_links.epic_key
        and not knowledge.traceability_links.parent_key
        and not knowledge.traceability_links.related_issues
        and not knowledge.traceability_links.blocked_by
        and not knowledge.traceability_links.blocks
        and not knowledge.traceability_links.depends_on
    )
    if is_orphan:
        findings.append(
            ValidationIssue(
                issue_key=key,
                severity="info",
                category="orphan",
                message="Issue has no epic link and no traceability links.",
                suggestion="Link this issue to its parent epic or related issues.",
            )
        )

    # 7. Missing context flagged by extraction
    if knowledge.missing_context:
        for gap in knowledge.missing_context:
            findings.append(
                ValidationIssue(
                    issue_key=key,
                    severity="info",
                    category="missing_context",
                    message=f"AI identified a context gap: {gap}",
                )
            )

    # 8. Open questions remain
    if knowledge.open_questions:
        findings.append(
            ValidationIssue(
                issue_key=key,
                severity="info",
                category="open_questions",
                message=f"{len(knowledge.open_questions)} open question(s) remain unresolved.",
                suggestion="Address open questions before implementation begins.",
            )
        )

    # Build result
    completeness = calculate_completeness_score(knowledge)

    return IssueValidationResult(
        issue_key=key,
        completeness_score=completeness,
        confidence_score=knowledge.confidence_score,
        issues_found=findings,
        # is_valid is auto-derived by the model validator
    )


# ── Completeness scoring ─────────────────────────────────────────────────

# Each field contributes a weight to the 0.0–1.0 completeness score.
_FIELD_WEIGHTS: list[tuple[str, float]] = [
    ("business_objective", 0.15),
    ("scope", 0.10),
    ("functional_requirements", 0.15),
    ("non_functional_requirements", 0.05),
    ("acceptance_criteria", 0.20),
    ("business_rules", 0.05),
    ("constraints", 0.05),
    ("dependencies", 0.05),
    ("traceability_epic", 0.10),
    ("timeline", 0.05),
    ("risks_and_assumptions", 0.05),
]


def calculate_completeness_score(knowledge: StructuredIssueKnowledge) -> float:
    """Compute a weighted completeness score for a single issue.

    The score ranges from 0.0 (nothing populated) to 1.0 (all fields
    richly populated).  Field weights are defined in ``_FIELD_WEIGHTS``.

    Args:
        knowledge: Structured knowledge for one issue.

    Returns:
        Completeness score between 0.0 and 1.0 (rounded to 2 decimals).
    """
    score = 0.0

    for field_name, weight in _FIELD_WEIGHTS:
        if field_name == "traceability_epic":
            # Traceability: any link present
            tl = knowledge.traceability_links
            has_links = bool(
                tl.epic_key
                or tl.parent_key
                or tl.related_issues
                or tl.blocked_by
                or tl.blocks
                or tl.depends_on
            )
            if has_links:
                score += weight
            continue

        if field_name == "timeline":
            if knowledge.timeline is not None:
                score += weight
            continue

        value = getattr(knowledge, field_name, None)
        if value:
            # String fields: reward longer content
            if isinstance(value, str) and len(value) > 10:
                score += weight
            elif isinstance(value, str) and value:
                score += weight * 0.5
            # List fields: reward having at least 2 items
            elif isinstance(value, list) and len(value) >= 2:
                score += weight
            elif isinstance(value, list) and len(value) == 1:
                score += weight * 0.7
            # Other truthy values
            elif not isinstance(value, (str, list)):
                score += weight

    return round(min(score, 1.0), 2)


# ── Project-wide quality report ──────────────────────────────────────────


def generate_quality_report(results: list[IssueValidationResult]) -> QualityReport:
    """Roll up per-issue validation results into a project-wide report.

    The overall quality score is the mean completeness score penalised by
    the proportion of critical findings.  The letter grade is derived
    from this score.

    Args:
        results: List of ``IssueValidationResult`` objects.

    Returns:
        A ``QualityReport`` with counts, scores, and an executive summary.
    """
    if not results:
        return QualityReport(
            generated_at=_now_iso(),
            total_issues_analyzed=0,
            summary="No issues were analysed.",
        )

    total = len(results)
    all_findings = [f for r in results for f in r.issues_found]
    critical = sum(1 for f in all_findings if f.severity == "critical")
    warnings = sum(1 for f in all_findings if f.severity == "warning")
    infos = sum(1 for f in all_findings if f.severity == "info")

    avg_completeness = sum(r.completeness_score for r in results) / total
    avg_confidence = sum(r.confidence_score for r in results) / total

    # Penalise by proportion of issues with critical findings
    issues_with_critical = sum(1 for r in results if not r.is_valid)
    penalty = (issues_with_critical / total) * 0.3  # up to 30% penalty
    quality_score = round(max(avg_completeness - penalty, 0.0), 2)

    grade = _score_to_grade(quality_score)

    # Build executive summary
    summary_lines = [
        f"Analysed {total} issue(s).",
        f"Overall quality score: {quality_score:.0%} (grade {grade}).",
        f"Average completeness: {avg_completeness:.0%}.",
        f"Average confidence: {avg_confidence:.0%}.",
        f"Findings: {critical} critical, {warnings} warning(s), {infos} info.",
    ]
    if issues_with_critical:
        summary_lines.append(
            f"{issues_with_critical} issue(s) have critical quality gaps."
        )

    return QualityReport(
        generated_at=_now_iso(),
        total_issues_analyzed=total,
        total_validation_issues=len(all_findings),
        critical_count=critical,
        warning_count=warnings,
        info_count=infos,
        overall_quality_score=quality_score,
        issue_results=results,
        summary="\n".join(summary_lines),
    )


# ── Private helpers ──────────────────────────────────────────────────────


def _score_to_grade(score: float) -> str:
    """Convert a 0.0–1.0 quality score to a letter grade."""
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.55:
        return "C"
    if score >= 0.40:
        return "D"
    return "F"


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
