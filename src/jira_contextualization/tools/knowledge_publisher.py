"""Knowledge publisher.

Serialises ``StructuredIssueKnowledge``, ``ProjectKnowledge``, and
``QualityReport`` objects to JSON and human-readable Markdown files.
Every public function returns the **absolute path** of the file it
wrote.

Public API
----------
- ``publish_issue_json(knowledge, output_dir)``
- ``publish_issue_markdown(knowledge, output_dir)``
- ``publish_project_knowledge(project, output_dir)``
- ``publish_validation_report(report, output_dir)``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from textwrap import dedent

from jira_contextualization.models.project_knowledge import ProjectKnowledge
from jira_contextualization.models.structured_knowledge import (
    StructuredIssueKnowledge,
)
from jira_contextualization.models.validation_report import (
    QualityReport,
)

logger = logging.getLogger(__name__)


# ── Per-issue JSON ───────────────────────────────────────────────────────


def publish_issue_json(
    knowledge: StructuredIssueKnowledge,
    output_dir: str,
) -> str:
    """Write a single issue's knowledge to a JSON file.

    The file is named ``<issue_key>.json`` (e.g. ``PROJ-42.json``).

    Args:
        knowledge: Structured knowledge for one issue.
        output_dir: Directory to write the file into.

    Returns:
        Absolute path to the written JSON file.
    """
    out = _ensure_dir(output_dir)
    filename = f"{knowledge.issue_key}.json"
    filepath = out / filename

    data = knowledge.model_dump(mode="json")
    filepath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote issue JSON: %s", filepath)
    return str(filepath.resolve())


# ── Per-issue Markdown ───────────────────────────────────────────────────


def publish_issue_markdown(
    knowledge: StructuredIssueKnowledge,
    output_dir: str,
) -> str:
    """Write a human-readable Markdown report for a single issue.

    Sections: Summary, Business Objective, Requirements, Acceptance
    Criteria, Dependencies, Risks & Assumptions, Quality Score.

    Args:
        knowledge: Structured knowledge for one issue.
        output_dir: Directory to write the file into.

    Returns:
        Absolute path to the written Markdown file.
    """
    out = _ensure_dir(output_dir)
    filename = f"{knowledge.issue_key}.md"
    filepath = out / filename

    lines: list[str] = []
    k = knowledge  # shorthand

    # ── Header ───────────────────────────────────────────────────────
    lines.append(f"# {k.issue_key}: {k.summary}")
    lines.append("")

    # ── Business Objective ───────────────────────────────────────────
    lines.append("## Business Objective")
    lines.append("")
    lines.append(k.business_objective or "_Not identified._")
    lines.append("")

    # ── Scope ────────────────────────────────────────────────────────
    if k.scope:
        lines.append("## Scope")
        lines.append("")
        lines.append(k.scope)
        lines.append("")

    # ── Requirements ─────────────────────────────────────────────────
    lines.append("## Requirements")
    lines.append("")

    if k.functional_requirements:
        lines.append("### Functional Requirements")
        lines.append("")
        for req in k.functional_requirements:
            lines.append(f"- {req}")
        lines.append("")

    if k.non_functional_requirements:
        lines.append("### Non-Functional Requirements")
        lines.append("")
        for req in k.non_functional_requirements:
            lines.append(f"- {req}")
        lines.append("")

    if not k.functional_requirements and not k.non_functional_requirements:
        lines.append("_No requirements extracted._")
        lines.append("")

    # ── Acceptance Criteria ──────────────────────────────────────────
    lines.append("## Acceptance Criteria")
    lines.append("")

    if k.acceptance_criteria:
        for ac in k.acceptance_criteria:
            lines.append(f"### {ac.id}: {ac.description}")
            lines.append("")
            if ac.given or ac.when or ac.then:
                if ac.given:
                    lines.append(f"- **Given** {ac.given}")
                if ac.when:
                    lines.append(f"- **When** {ac.when}")
                if ac.then:
                    lines.append(f"- **Then** {ac.then}")
                lines.append("")
            testable = "✅ Testable" if ac.is_testable else "⚠️ Not directly testable"
            lines.append(f"_{testable}_")
            lines.append("")
    else:
        lines.append("_No acceptance criteria found._")
        lines.append("")

    # ── Business Rules ───────────────────────────────────────────────
    if k.business_rules:
        lines.append("## Business Rules")
        lines.append("")
        for rule in k.business_rules:
            lines.append(f"- {rule}")
        lines.append("")

    # ── Constraints ──────────────────────────────────────────────────
    if k.constraints:
        lines.append("## Constraints")
        lines.append("")
        for c in k.constraints:
            lines.append(f"- {c}")
        lines.append("")

    # ── Dependencies ─────────────────────────────────────────────────
    lines.append("## Dependencies")
    lines.append("")

    tl = k.traceability_links
    dep_items: list[str] = []
    if tl.epic_key:
        dep_items.append(f"- **Epic**: {tl.epic_key}")
    if tl.parent_key:
        dep_items.append(f"- **Parent**: {tl.parent_key}")
    if tl.blocked_by:
        dep_items.append(f"- **Blocked by**: {', '.join(tl.blocked_by)}")
    if tl.blocks:
        dep_items.append(f"- **Blocks**: {', '.join(tl.blocks)}")
    if tl.depends_on:
        dep_items.append(f"- **Depends on**: {', '.join(tl.depends_on)}")
    if tl.related_issues:
        dep_items.append(f"- **Related**: {', '.join(tl.related_issues)}")
    if tl.cloned_from:
        dep_items.append(f"- **Cloned from**: {', '.join(tl.cloned_from)}")

    if dep_items:
        lines.extend(dep_items)
    else:
        lines.append("_No dependencies identified._")
    lines.append("")

    # ── Risks & Assumptions ──────────────────────────────────────────
    if k.risks_and_assumptions:
        lines.append("## Risks & Assumptions")
        lines.append("")
        for item in k.risks_and_assumptions:
            lines.append(f"- {item}")
        lines.append("")

    # ── Open Questions ───────────────────────────────────────────────
    if k.open_questions:
        lines.append("## Open Questions")
        lines.append("")
        for q in k.open_questions:
            lines.append(f"- {q}")
        lines.append("")

    # ── Quality Score ────────────────────────────────────────────────
    lines.append("## Quality Score")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Completeness | {k.completeness_score:.0%} |")
    lines.append(f"| Confidence | {k.confidence_score:.0%} |")
    lines.append("")

    if k.missing_context:
        lines.append("### Information Gaps")
        lines.append("")
        for gap in k.missing_context:
            lines.append(f"- {gap}")
        lines.append("")

    # ── Extraction Notes ─────────────────────────────────────────────
    if k.extraction_notes:
        lines.append("## Extraction Notes")
        lines.append("")
        for note in k.extraction_notes:
            lines.append(f"- {note}")
        lines.append("")

    content = "\n".join(lines)
    filepath.write_text(content, encoding="utf-8")
    logger.info("Wrote issue Markdown: %s", filepath)
    return str(filepath.resolve())


# ── Project-level knowledge ──────────────────────────────────────────────


def publish_project_knowledge(
    project: ProjectKnowledge,
    output_dir: str,
) -> dict[str, str]:
    """Write consolidated project knowledge as JSON and executive summary MD.

    Files written:
    - ``<project_key>_knowledge.json`` — full structured data.
    - ``<project_key>_summary.md``     — human-readable executive summary.

    Args:
        project: Full project knowledge model.
        output_dir: Directory to write files into.

    Returns:
        ``{'json': '<path>', 'markdown': '<path>'}``
    """
    out = _ensure_dir(output_dir)
    pk = project.project_key

    # ── JSON ─────────────────────────────────────────────────────────
    json_path = out / f"{pk}_knowledge.json"
    data = project.model_dump(mode="json")
    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Executive summary Markdown ───────────────────────────────────
    md_path = out / f"{pk}_summary.md"
    md_lines: list[str] = []

    md_lines.append(f"# {project.project_name} — Knowledge Summary")
    md_lines.append("")
    md_lines.append(f"**Project key:** {pk}")
    md_lines.append(f"**Generated:** {project.generated_at}")
    md_lines.append(f"**Total issues:** {project.total_issues}")
    md_lines.append("")

    # Quality metrics
    qm = project.quality_metrics
    md_lines.append("## Quality Overview")
    md_lines.append("")
    md_lines.append(f"| Metric | Value |")
    md_lines.append(f"|--------|-------|")
    md_lines.append(f"| Quality grade | **{qm.quality_grade}** |")
    md_lines.append(f"| Avg. confidence | {qm.avg_confidence_score:.0%} |")
    md_lines.append(f"| Avg. completeness | {qm.avg_completeness_score:.0%} |")
    md_lines.append(f"| Issues with AC | {qm.issues_with_ac} / {qm.total_issues} |")
    md_lines.append(f"| Issues with links | {qm.issues_with_links} / {qm.total_issues} |")
    md_lines.append(f"| Missing priority | {qm.issues_missing_priority} |")
    md_lines.append(f"| Missing description | {qm.issues_missing_description} |")
    md_lines.append("")

    # Epics
    if project.epics:
        md_lines.append("## Epics")
        md_lines.append("")
        md_lines.append("| Epic | Stories | Avg Confidence | Components |")
        md_lines.append("|------|---------|----------------|------------|")
        for epic in project.epics:
            title = epic.title or epic.epic_key
            comps = ", ".join(epic.components[:3]) or "—"
            md_lines.append(
                f"| {title} | {epic.story_count} "
                f"| {epic.avg_confidence:.0%} | {comps} |"
            )
        md_lines.append("")

    # Components
    if project.components:
        md_lines.append("## Components")
        md_lines.append("")
        md_lines.append("| Component | Issues | Epics |")
        md_lines.append("|-----------|--------|-------|")
        for comp in project.components:
            epics_str = ", ".join(comp.epics[:3]) or "—"
            md_lines.append(
                f"| {comp.name} | {comp.story_count} | {epics_str} |"
            )
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Wrote project knowledge: %s, %s", json_path, md_path)
    return {
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
    }


# ── Validation report ────────────────────────────────────────────────────


def publish_validation_report(
    report: QualityReport,
    output_dir: str,
) -> str:
    """Write a validation report as a Markdown file with severity tables.

    Args:
        report: Quality report produced by the validator.
        output_dir: Directory to write the file into.

    Returns:
        Absolute path to the written Markdown file.
    """
    out = _ensure_dir(output_dir)
    filepath = out / "validation_report.md"

    lines: list[str] = []

    lines.append("# Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {report.generated_at}")
    lines.append(f"**Issues analysed:** {report.total_issues_analyzed}")
    lines.append(f"**Overall quality score:** {report.overall_quality_score:.0%}")
    lines.append("")

    # Summary
    if report.summary:
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(report.summary)
        lines.append("")

    # Severity breakdown
    lines.append("## Severity Breakdown")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    lines.append(f"| 🔴 Critical | {report.critical_count} |")
    lines.append(f"| 🟡 Warning | {report.warning_count} |")
    lines.append(f"| 🔵 Info | {report.info_count} |")
    lines.append(f"| **Total** | **{report.total_validation_issues}** |")
    lines.append("")

    # Per-issue details
    if report.issue_results:
        lines.append("## Per-Issue Results")
        lines.append("")

        # Summary table
        lines.append("| Issue | Valid | Completeness | Confidence | Findings |")
        lines.append("|-------|-------|--------------|------------|----------|")
        for result in report.issue_results:
            valid = "✅" if result.is_valid else "❌"
            findings_count = len(result.issues_found)
            lines.append(
                f"| {result.issue_key} | {valid} "
                f"| {result.completeness_score:.0%} "
                f"| {result.confidence_score:.0%} "
                f"| {findings_count} |"
            )
        lines.append("")

        # Detailed findings for issues with problems
        issues_with_findings = [
            r for r in report.issue_results if r.issues_found
        ]
        if issues_with_findings:
            lines.append("## Detailed Findings")
            lines.append("")

            for result in issues_with_findings:
                lines.append(f"### {result.issue_key}")
                lines.append("")
                lines.append("| Severity | Category | Message | Suggestion |")
                lines.append("|----------|----------|---------|------------|")
                for finding in result.issues_found:
                    sev_icon = {
                        "critical": "🔴",
                        "warning": "🟡",
                        "info": "🔵",
                    }.get(finding.severity, "⚪")
                    suggestion = finding.suggestion or "—"
                    lines.append(
                        f"| {sev_icon} {finding.severity} "
                        f"| {finding.category} "
                        f"| {finding.message} "
                        f"| {suggestion} |"
                    )
                lines.append("")

    content = "\n".join(lines)
    filepath.write_text(content, encoding="utf-8")
    logger.info("Wrote validation report: %s", filepath)
    return str(filepath.resolve())


# ── Private helpers ──────────────────────────────────────────────────────


def _ensure_dir(path: str) -> Path:
    """Create the directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
