"""
Jira Contextualization Flow — Orchestrates the 6-stage knowledge build pipeline.

Pipeline Stages (Refined Architecture):
  1. Ingest & Normalize (deterministic)
  2. Extract Requirements (CrewAI — ExtractionCrew × DeepSeek Chat)
  3. Build Relationships (deterministic)
  4. Consolidate & Enrich (CrewAI — ConsolidationCrew × DeepSeek Reasoner)  [NEW]
  5. Validate Knowledge (CrewAI — ValidationCrew × DeepSeek Chat)
  6. Publish Artifacts (deterministic)

Changes from v1:
  - Stages 2/4/5 now use proper @CrewBase crews instead of raw llm.call()
  - New Stage 4 (Consolidation) for dedup + project-level enrichment
  - Retry/backoff via llm_utils on all LLM calls
  - All LLM calls use DeepSeek (Chat + Reasoner) to avoid Gemini quotas
  - Caching preserved for every LLM stage (2, 4, 5)
"""

from __future__ import annotations

import builtins
import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

# Force unbuffered print on Windows
print = partial(builtins.print, flush=True)  # type: ignore[assignment]

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from jira_contextualization.models.normalized_issue import NormalizedIssue
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
from jira_contextualization.tools.csv_parser import parse_jira_csv
from jira_contextualization.tools.knowledge_publisher import (
    publish_issue_json,
    publish_issue_markdown,
    publish_project_knowledge,
    publish_validation_report,
)
from jira_contextualization.tools.knowledge_validator import (
    calculate_completeness_score,
    generate_quality_report,
    validate_issue,
)
from jira_contextualization.tools.relationship_builder import (
    build_dependency_graph,
    build_epic_hierarchy,
    find_related_clusters,
    group_by_component,
)
from jira_contextualization.tools.requirement_extractor import (
    extract_requirements_deterministic,
)
from jira_contextualization.tools.wiki_markup_parser import (
    extract_sections,
    parse_jira_markup,
)
from jira_contextualization.tools.llm_utils import (
    get_deepseek_llm,
    get_deepseek_reasoner_llm,
    parse_llm_json,
    safe_llm_extract,
)

load_dotenv()


# ─── Flow State ──────────────────────────────────────────────────────────────


class PipelineState(BaseModel):
    """State passed between pipeline stages."""

    csv_path: str = ""
    output_dir: str = "output"
    normalized_issues: list[dict] = Field(default_factory=list)
    knowledge_objects: list[dict] = Field(default_factory=list)
    relationships: dict = Field(default_factory=dict)
    consolidation_result: dict = Field(default_factory=dict)    # NEW
    validation_results: list[dict] = Field(default_factory=list)
    quality_report: dict = Field(default_factory=dict)
    published_files: list[str] = Field(default_factory=list)
    stage_timings: dict = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


# ─── Crew-Based Extraction ──────────────────────────────────────────────────


def _extract_with_crew(issue: NormalizedIssue, deterministic: dict) -> dict:
    """Use ExtractionCrew to extract knowledge from a single issue.

    Falls back to safe_llm_extract if crew kickoff fails.
    """
    from jira_contextualization.crews.extraction_crew import ExtractionCrew

    try:
        inputs = {
            "issue_key": issue.issue_key,
            "summary": issue.summary,
            "status": issue.status,
            "priority": issue.priority,
            "components": ", ".join(issue.components),
            "description": issue.description[:4000],
            "det_ac_count": str(len(deterministic.get("acceptance_criteria", []))),
            "det_story_count": str(len(deterministic.get("user_stories", []))),
            "det_gwt_count": str(len(deterministic.get("given_when_then", []))),
        }
        result = ExtractionCrew().crew().kickoff(inputs=inputs)
        raw_text = result.raw if hasattr(result, "raw") else str(result)
        return parse_llm_json(raw_text)
    except Exception as e:
        print(f"    ⚠️  Crew extraction failed for {issue.issue_key}, falling back to direct LLM: {str(e)[:100]}")
        return _fallback_llm_extract(issue, deterministic)


def _fallback_llm_extract(issue: NormalizedIssue, deterministic: dict) -> dict:
    """Fallback: direct LLM call with retry if crew fails."""
    llm = get_deepseek_llm()
    prompt = f"""You are a senior business analyst. Extract structured requirements from this Jira ticket.

## Ticket: {issue.issue_key}
**Summary**: {issue.summary}
**Status**: {issue.status} | **Priority**: {issue.priority}
**Components**: {', '.join(issue.components)}

## Description:
{issue.description[:3000]}

## Instructions:
Return a JSON object with EXACTLY these fields:
{{
  "business_objective": "one sentence describing the business goal",
  "scope": "what is in/out of scope",
  "functional_requirements": ["list of functional requirements"],
  "non_functional_requirements": ["list of NFRs if any"],
  "acceptance_criteria": [
    {{"id": "AC-1", "description": "criterion text", "given": "given clause or null", "when": "when clause or null", "then": "then clause or null", "is_testable": true}}
  ],
  "business_rules": ["list of business rules"],
  "constraints": ["list of constraints"],
  "risks_and_assumptions": ["list of risks and assumptions"],
  "decisions": ["list of design decisions"],
  "open_questions": ["list of unresolved questions"]
}}

IMPORTANT: Return ONLY valid JSON. No markdown, no explanation."""

    return safe_llm_extract(llm, prompt, fallback={
        "business_objective": issue.summary,
        "scope": "",
        "functional_requirements": deterministic.get("user_stories", []),
        "non_functional_requirements": [],
        "acceptance_criteria": [
            {"id": f"AC-{i+1}", "description": ac, "given": None, "when": None, "then": None, "is_testable": True}
            for i, ac in enumerate(deterministic.get("acceptance_criteria", []))
        ],
        "business_rules": deterministic.get("business_rules", []),
        "constraints": deterministic.get("constraints", []),
        "risks_and_assumptions": [],
        "decisions": [],
        "open_questions": [],
    })


# ─── Crew-Based Validation ──────────────────────────────────────────────────


def _validate_with_crew(knowledge: StructuredIssueKnowledge) -> dict:
    """Use ValidationCrew to detect ambiguity and quality issues."""
    from jira_contextualization.crews.validation_crew import ValidationCrew

    try:
        inputs = {
            "issue_key": knowledge.issue_key,
            "summary": knowledge.summary,
            "business_objective": knowledge.business_objective,
            "functional_requirements": json.dumps(knowledge.functional_requirements[:5]),
            "ac_count": str(len(knowledge.acceptance_criteria)),
            "rule_confidence": str(round(knowledge.confidence_score, 2)),
        }
        result = ValidationCrew().crew().kickoff(inputs=inputs)
        raw_text = result.raw if hasattr(result, "raw") else str(result)
        return parse_llm_json(raw_text)
    except Exception as e:
        print(f"    ⚠️  Crew validation failed for {knowledge.issue_key}, falling back: {str(e)[:100]}")
        return _fallback_llm_validate(knowledge)


def _fallback_llm_validate(knowledge: StructuredIssueKnowledge) -> dict:
    """Fallback: direct LLM validation with retry."""
    llm = get_deepseek_llm()
    prompt = f"""You are a QA specialist. Review this extracted knowledge for quality issues.

## Ticket: {knowledge.issue_key}
**Summary**: {knowledge.summary}
**Business Objective**: {knowledge.business_objective}
**Functional Requirements**: {json.dumps(knowledge.functional_requirements[:5])}
**Acceptance Criteria**: {len(knowledge.acceptance_criteria)} items

## Instructions:
Check for:
1. Ambiguous language ("should", "might", "possibly", "as needed")
2. Untestable acceptance criteria
3. Missing critical information
4. Conflicting requirements

Return a JSON object:
{{
  "issues": [
    {{"severity": "critical|warning|info", "category": "ambiguity|missing_ac|untestable|conflict|incomplete", "message": "description", "suggestion": "how to fix"}}
  ],
  "adjusted_confidence": 0.85
}}

Return ONLY valid JSON."""

    return safe_llm_extract(llm, prompt, fallback={
        "issues": [],
        "adjusted_confidence": knowledge.confidence_score,
    })


# ─── Consolidation ──────────────────────────────────────────────────────────


def _run_consolidation_crew(
    knowledge_objects: list[StructuredIssueKnowledge],
    relationships: dict,
) -> dict:
    """Use ConsolidationCrew for cross-issue dedup + project enrichment."""
    from jira_contextualization.crews.consolidation_crew import ConsolidationCrew

    # Build compact summaries for context (avoid token limits)
    summaries = []
    for k in knowledge_objects:
        summaries.append(
            f"- {k.issue_key}: {k.summary} | Obj: {k.business_objective[:80]} | "
            f"FR: {len(k.functional_requirements)} | AC: {len(k.acceptance_criteria)}"
        )
    issue_summaries_text = "\n".join(summaries)

    epic_hierarchy = relationships.get("epic_hierarchy", {})
    comp_groups = relationships.get("component_groups", {})
    project_name = knowledge_objects[0].issue_key.split("-")[0] if knowledge_objects else "UNKNOWN"

    inputs = {
        "total_issues": str(len(knowledge_objects)),
        "issue_summaries": issue_summaries_text[:12000],  # Token limit safety
        "project_name": project_name,
        "components": ", ".join(comp_groups.keys()),
        "epic_count": str(len(epic_hierarchy)),
    }

    try:
        result = ConsolidationCrew().crew().kickoff(inputs=inputs)
        raw_text = result.raw if hasattr(result, "raw") else str(result)
        return parse_llm_json(raw_text)
    except Exception as e:
        print(f"    ⚠️  Consolidation crew failed, using fallback: {str(e)[:150]}")
        return _fallback_consolidation(knowledge_objects, relationships)


def _fallback_consolidation(
    knowledge_objects: list[StructuredIssueKnowledge],
    relationships: dict,
) -> dict:
    """Fallback consolidation using direct LLM call."""
    llm = get_deepseek_llm()
    summaries = [
        f"- {k.issue_key}: {k.summary}"
        for k in knowledge_objects[:100]  # Limit for token safety
    ]

    prompt = f"""You are a principal architect. Analyze these {len(knowledge_objects)} Jira issues and generate project-level insights.

Issues:
{chr(10).join(summaries)}

Return a JSON object:
{{
  "project_summary": "2-3 paragraph summary of the project",
  "domain_groups": [{{"domain_name": "name", "description": "desc", "issue_keys": ["KEY-1"], "key_capabilities": ["cap1"]}}],
  "key_themes": ["theme1", "theme2"],
  "cross_cutting_concerns": ["concern1"],
  "capability_map": {{"capability": ["KEY-1", "KEY-2"]}},
  "duplicate_requirements": []
}}

Return ONLY valid JSON."""

    return safe_llm_extract(llm, prompt, fallback={
        "project_summary": f"Project with {len(knowledge_objects)} issues.",
        "domain_groups": [],
        "key_themes": [],
        "cross_cutting_concerns": [],
        "capability_map": {},
        "duplicate_requirements": [],
    })


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _to_adjacency_list(dep_graph: dict) -> dict[str, list[str]]:
    """Convert a full dependency graph dict to adjacency list format."""
    adj: dict[str, list[str]] = {}
    edges = dep_graph.get("edges", [])
    if isinstance(edges, list):
        for edge in edges:
            if isinstance(edge, dict):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                if src and tgt:
                    adj.setdefault(src, []).append(tgt)
    return adj


# ─── Main Pipeline Flow ─────────────────────────────────────────────────────


class JiraContextualizationFlow(Flow[PipelineState]):
    """Main pipeline flow orchestrating all 6 stages."""

    @start()
    def stage_1_ingest(self) -> str:
        """Stage 1: Parse CSV and normalize all tickets."""
        print("\n" + "=" * 70)
        print("  STAGE 1: INGESTION & NORMALIZATION")
        print("=" * 70)
        start_time = time.time()

        csv_path = self.state.csv_path
        if not csv_path or not Path(csv_path).exists():
            default = Path(__file__).parent.parent.parent / "data" / "raw"
            csv_files = list(default.glob("*.csv"))
            if csv_files:
                csv_path = str(csv_files[0])
            else:
                self.state.errors.append("No CSV file found")
                return "error"

        print(f"  📂 Parsing: {Path(csv_path).name}")
        issues = parse_jira_csv(csv_path)
        print(f"  ✅ Parsed {len(issues)} tickets")

        self.state.normalized_issues = [iss.model_dump() for iss in issues]

        statuses = {}
        priorities = {}
        for iss in issues:
            statuses[iss.status] = statuses.get(iss.status, 0) + 1
            priorities[iss.priority] = priorities.get(iss.priority, 0) + 1

        print(f"  📊 Statuses: {statuses}")
        print(f"  📊 Priorities: {priorities}")

        elapsed = time.time() - start_time
        self.state.stage_timings["ingestion"] = round(elapsed, 2)
        print(f"  ⏱️  Completed in {elapsed:.2f}s")
        return "ingested"

    @listen(stage_1_ingest)
    def stage_2_extract(self, _: str) -> str:
        """Stage 2: Extract requirements using ExtractionCrew + deterministic."""
        print("\n" + "=" * 70)
        print("  STAGE 2: KNOWLEDGE EXTRACTION (ExtractionCrew × DeepSeek)")
        print("=" * 70)
        start_time = time.time()

        cache_path = Path(self.state.output_dir) / "knowledge_objects_cache.json"
        if cache_path.exists():
            print(f"  ✨ Found cached knowledge objects. Loading...")
            with open(cache_path, "r", encoding="utf-8") as f:
                self.state.knowledge_objects = json.load(f)
            print(f"  ✅ Loaded {len(self.state.knowledge_objects)} cached knowledge objects.")
            elapsed = time.time() - start_time
            self.state.stage_timings["extraction"] = round(elapsed, 2)
            return "extracted"

        issues = [NormalizedIssue(**d) for d in self.state.normalized_issues]
        knowledge_objects = []

        for i, issue in enumerate(issues):
            print(f"  [{i+1}/{len(issues)}] Extracting: {issue.issue_key} — {issue.summary[:60]}...")

            # Step 1: Deterministic extraction
            det_results = extract_requirements_deterministic(issue)

            # Step 2: Crew-based extraction (with fallback)
            llm_results = _extract_with_crew(issue, det_results)

            # Step 3: Merge into StructuredIssueKnowledge
            ac_list = []
            llm_acs = llm_results.get("acceptance_criteria", [])
            if isinstance(llm_acs, list):
                for j, ac in enumerate(llm_acs):
                    if isinstance(ac, dict):
                        ac_list.append(AcceptanceCriterion(
                            id=ac.get("id", f"AC-{j+1}"),
                            description=ac.get("description", ""),
                            given=ac.get("given"),
                            when=ac.get("when"),
                            then=ac.get("then"),
                            is_testable=ac.get("is_testable", True),
                        ))

            # Build traceability from issue links
            related = []
            blocked_by = []
            blocks = []
            depends_on = []
            cloned_from = []
            for link in issue.issue_links:
                if link.link_type == "Relates":
                    related.append(link.target_key)
                elif link.link_type == "Blocks" and link.direction == "inward":
                    blocked_by.append(link.target_key)
                elif link.link_type == "Blocks" and link.direction == "outward":
                    blocks.append(link.target_key)
                elif link.link_type == "Depends" and link.direction == "inward":
                    depends_on.append(link.target_key)
                elif link.link_type == "Cloners":
                    cloned_from.append(link.target_key)

            deps = []
            for link in issue.issue_links:
                deps.append(Dependency(
                    target_key=link.target_key,
                    dependency_type=link.link_type,
                    direction=link.direction,
                    description=None,
                ))

            knowledge = StructuredIssueKnowledge(
                issue_key=issue.issue_key,
                summary=issue.summary,
                business_objective=llm_results.get("business_objective", issue.summary),
                scope=llm_results.get("scope", ""),
                functional_requirements=llm_results.get("functional_requirements", []),
                non_functional_requirements=llm_results.get("non_functional_requirements", []),
                acceptance_criteria=ac_list,
                business_rules=llm_results.get("business_rules", []),
                constraints=llm_results.get("constraints", []),
                risks_and_assumptions=llm_results.get("risks_and_assumptions", []),
                dependencies=deps,
                decisions=llm_results.get("decisions", []),
                open_questions=llm_results.get("open_questions", []),
                traceability_links=TraceabilityLinks(
                    epic_key=issue.epic_link,
                    parent_key=issue.parent_link,
                    related_issues=related,
                    blocked_by=blocked_by,
                    blocks=blocks,
                    depends_on=depends_on,
                    cloned_from=cloned_from,
                ),
                timeline=Timeline(
                    created=issue.created,
                    updated=issue.updated,
                    resolved=issue.resolved,
                    sprints=issue.sprints,
                    status_history=[issue.status],
                ),
                extraction_notes=llm_results.get("_extraction_error", None) and [
                    f"LLM extraction error: {llm_results['_extraction_error']}"
                ] or [],
            )

            knowledge.completeness_score = calculate_completeness_score(knowledge)
            knowledge.confidence_score = min(
                1.0,
                0.3
                + (0.2 if knowledge.business_objective != issue.summary else 0)
                + (0.2 if len(knowledge.acceptance_criteria) > 0 else 0)
                + (0.15 if len(knowledge.functional_requirements) > 0 else 0)
                + (0.15 if issue.priority != "Unspecified" else 0),
            )

            knowledge_objects.append(knowledge)

        self.state.knowledge_objects = [k.model_dump() for k in knowledge_objects]

        # Save cache
        os.makedirs(self.state.output_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.state.knowledge_objects, f, indent=2, ensure_ascii=False)
        print(f"  💾 Cached extracted knowledge to {cache_path}")

        avg_conf = sum(k.confidence_score for k in knowledge_objects) / len(knowledge_objects) if knowledge_objects else 0
        avg_comp = sum(k.completeness_score for k in knowledge_objects) / len(knowledge_objects) if knowledge_objects else 0
        print(f"\n  ✅ Extracted knowledge for {len(knowledge_objects)} tickets")
        print(f"  📊 Avg Confidence: {avg_conf:.2f} | Avg Completeness: {avg_comp:.2f}")

        elapsed = time.time() - start_time
        self.state.stage_timings["extraction"] = round(elapsed, 2)
        print(f"  ⏱️  Completed in {elapsed:.2f}s")
        return "extracted"

    @listen(stage_2_extract)
    def stage_3_relationships(self, _: str) -> str:
        """Stage 3: Build relationship graphs and hierarchy."""
        print("\n" + "=" * 70)
        print("  STAGE 3: RELATIONSHIP & HIERARCHY BUILDING")
        print("=" * 70)
        start_time = time.time()

        issues = [NormalizedIssue(**d) for d in self.state.normalized_issues]

        epic_hierarchy = build_epic_hierarchy(issues)
        dep_graph = build_dependency_graph(issues)
        component_groups = group_by_component(issues)
        related_clusters = find_related_clusters(issues)

        self.state.relationships = {
            "epic_hierarchy": epic_hierarchy,
            "dependency_graph": dep_graph,
            "component_groups": component_groups,
            "related_clusters": related_clusters,
        }

        print(f"  ✅ Epic hierarchy: {len(epic_hierarchy)} epics")
        print(f"  ✅ Dependency graph: {len(dep_graph.get('nodes', []))} nodes, {len(dep_graph.get('edges', []))} edges")
        print(f"  ✅ Component groups: {list(component_groups.keys())}")
        print(f"  ✅ Related clusters: {len(related_clusters)} clusters")

        elapsed = time.time() - start_time
        self.state.stage_timings["relationships"] = round(elapsed, 2)
        print(f"  ⏱️  Completed in {elapsed:.2f}s")
        return "relationships_built"

    @listen(stage_3_relationships)
    def stage_4_consolidate(self, _: str) -> str:
        """Stage 4 [NEW]: Cross-issue consolidation & project enrichment."""
        print("\n" + "=" * 70)
        print("  STAGE 4: KNOWLEDGE CONSOLIDATION & ENRICHMENT (DeepSeek Reasoner)")
        print("=" * 70)
        start_time = time.time()

        cache_path = Path(self.state.output_dir) / "consolidation_cache.json"
        if cache_path.exists():
            print(f"  ✨ Found cached consolidation results. Loading...")
            with open(cache_path, "r", encoding="utf-8") as f:
                self.state.consolidation_result = json.load(f)
            print("  ✅ Loaded cached consolidation results.")
            elapsed = time.time() - start_time
            self.state.stage_timings["consolidation"] = round(elapsed, 2)
            return "consolidated"

        knowledge_objects = [
            StructuredIssueKnowledge(**d) for d in self.state.knowledge_objects
        ]

        print(f"  🔍 Analyzing {len(knowledge_objects)} issues for cross-issue consolidation...")
        consolidation = _run_consolidation_crew(knowledge_objects, self.state.relationships)

        self.state.consolidation_result = consolidation

        # Save cache
        os.makedirs(self.state.output_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(consolidation, f, indent=2, ensure_ascii=False, default=str)
        print(f"  💾 Cached consolidation results to {cache_path}")

        # Summary
        dupes = consolidation.get("duplicate_requirements", [])
        domains = consolidation.get("domain_groups", [])
        themes = consolidation.get("key_themes", [])
        print(f"\n  ✅ Consolidation complete")
        print(f"  📊 Duplicate pairs found: {len(dupes)}")
        print(f"  📊 Domain groups: {len(domains)}")
        print(f"  📊 Key themes: {len(themes)}")
        if themes:
            print(f"  📋 Themes: {', '.join(themes[:5])}...")

        elapsed = time.time() - start_time
        self.state.stage_timings["consolidation"] = round(elapsed, 2)
        print(f"  ⏱️  Completed in {elapsed:.2f}s")
        return "consolidated"

    @listen(stage_4_consolidate)
    def stage_5_validate(self, _: str) -> str:
        """Stage 5: Validate knowledge quality using rules + ValidationCrew."""
        print("\n" + "=" * 70)
        print("  STAGE 5: KNOWLEDGE VALIDATION (ValidationCrew × DeepSeek)")
        print("=" * 70)
        start_time = time.time()

        val_cache_path = Path(self.state.output_dir) / "validation_results_cache.json"
        rep_cache_path = Path(self.state.output_dir) / "quality_report_cache.json"

        if val_cache_path.exists() and rep_cache_path.exists():
            print(f"  ✨ Found cached validation results. Loading...")
            with open(val_cache_path, "r", encoding="utf-8") as f:
                self.state.validation_results = json.load(f)
            with open(rep_cache_path, "r", encoding="utf-8") as f:
                self.state.quality_report = json.load(f)

            val_map = {res["issue_key"]: res for res in self.state.validation_results}
            for k in self.state.knowledge_objects:
                if k["issue_key"] in val_map:
                    k["completeness_score"] = val_map[k["issue_key"]]["completeness_score"]
                    k["confidence_score"] = val_map[k["issue_key"]]["confidence_score"]

            print("  ✅ Loaded cached validation results.")
            elapsed = time.time() - start_time
            self.state.stage_timings["validation"] = round(elapsed, 2)
            return "validated"

        knowledge_objects = [
            StructuredIssueKnowledge(**d) for d in self.state.knowledge_objects
        ]
        all_results = []

        for i, knowledge in enumerate(knowledge_objects):
            print(f"  [{i+1}/{len(knowledge_objects)}] Validating: {knowledge.issue_key}...")

            # Step 1: Rule-based validation
            rule_result = validate_issue(knowledge)

            # Step 2: Crew-based validation (ambiguity, conflicts)
            llm_result = _validate_with_crew(knowledge)

            # Merge LLM issues into rule results
            llm_issues = llm_result.get("issues", [])
            if isinstance(llm_issues, list):
                for iss in llm_issues:
                    if isinstance(iss, dict):
                        rule_result.issues_found.append(ValidationIssue(
                            issue_key=knowledge.issue_key,
                            severity=iss.get("severity", "info"),
                            category=iss.get("category", "ambiguity"),
                            message=iss.get("message", ""),
                            suggestion=iss.get("suggestion"),
                        ))

            adjusted = llm_result.get("adjusted_confidence")
            if adjusted and isinstance(adjusted, (int, float)):
                rule_result.confidence_score = (
                    rule_result.confidence_score * 0.6 + float(adjusted) * 0.4
                )

            rule_result.is_valid = not any(
                i.severity == "critical" for i in rule_result.issues_found
            )
            all_results.append(rule_result)

        report = generate_quality_report(all_results)
        self.state.validation_results = [r.model_dump() for r in all_results]
        self.state.quality_report = report.model_dump()

        for knowledge_dict, result in zip(self.state.knowledge_objects, all_results):
            knowledge_dict["completeness_score"] = result.completeness_score
            knowledge_dict["confidence_score"] = result.confidence_score

        # Save cache
        os.makedirs(self.state.output_dir, exist_ok=True)
        with open(val_cache_path, "w", encoding="utf-8") as f:
            json.dump(self.state.validation_results, f, indent=2, ensure_ascii=False)
        with open(rep_cache_path, "w", encoding="utf-8") as f:
            json.dump(self.state.quality_report, f, indent=2, ensure_ascii=False)
        print(f"  💾 Cached validation results")

        print(f"\n  ✅ Validated {len(all_results)} tickets")
        print(f"  📊 Overall Quality: {report.overall_quality_score:.2f}")
        print(f"  🔴 Critical: {report.critical_count} | ⚠️  Warnings: {report.warning_count} | ℹ️  Info: {report.info_count}")

        elapsed = time.time() - start_time
        self.state.stage_timings["validation"] = round(elapsed, 2)
        print(f"  ⏱️  Completed in {elapsed:.2f}s")
        return "validated"

    @listen(stage_5_validate)
    def stage_6_publish(self, _: str) -> str:
        """Stage 6: Publish all output artifacts."""
        print("\n" + "=" * 70)
        print("  STAGE 6: KNOWLEDGE PUBLISHING")
        print("=" * 70)
        start_time = time.time()

        output_dir = Path(self.state.output_dir)
        knowledge_objects = [
            StructuredIssueKnowledge(**d) for d in self.state.knowledge_objects
        ]
        issues = [NormalizedIssue(**d) for d in self.state.normalized_issues]
        report = QualityReport(**self.state.quality_report)
        consolidation = self.state.consolidation_result
        published = []

        # 1. Per-issue JSON
        json_dir = str(output_dir / "knowledge_json")
        for k in knowledge_objects:
            path = publish_issue_json(k, json_dir)
            published.append(path)
        print(f"  ✅ Published {len(knowledge_objects)} JSON files")

        # 2. Per-issue Markdown
        md_dir = str(output_dir / "knowledge_markdown")
        for k in knowledge_objects:
            path = publish_issue_markdown(k, md_dir)
            published.append(path)
        print(f"  ✅ Published {len(knowledge_objects)} Markdown files")

        # 3. Relationship graph
        graph_path = str(output_dir / "relationship_graph.json")
        os.makedirs(str(output_dir), exist_ok=True)
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(self.state.relationships, f, indent=2, default=str)
        published.append(graph_path)
        print("  ✅ Published relationship_graph.json")

        # 4. Consolidation report (NEW)
        consol_path = str(output_dir / "consolidation_report.json")
        with open(consol_path, "w", encoding="utf-8") as f:
            json.dump(consolidation, f, indent=2, ensure_ascii=False, default=str)
        published.append(consol_path)
        print("  ✅ Published consolidation_report.json")

        # 5. Consolidated project knowledge
        epic_hierarchy = self.state.relationships.get("epic_hierarchy", {})
        epic_summaries = []
        for epic_key, story_keys in epic_hierarchy.items():
            epic_knowledges = [k for k in knowledge_objects if k.issue_key in story_keys]
            avg_conf = (
                sum(k.confidence_score for k in epic_knowledges) / len(epic_knowledges)
                if epic_knowledges else 0
            )
            comps = set()
            for sk in story_keys:
                iss = next((i for i in issues if i.issue_key == sk), None)
                if iss:
                    comps.update(iss.components)
            epic_summaries.append(EpicSummary(
                epic_key=epic_key,
                title=None,
                story_count=len(story_keys),
                stories=story_keys,
                components=list(comps),
                avg_confidence=round(avg_conf, 2),
            ))

        comp_groups = self.state.relationships.get("component_groups", {})
        comp_summaries = []
        for comp_name, story_keys in comp_groups.items():
            comp_epics = set()
            for sk in story_keys:
                iss = next((i for i in issues if i.issue_key == sk), None)
                if iss and iss.epic_link:
                    comp_epics.add(iss.epic_link)
            comp_summaries.append(ComponentSummary(
                name=comp_name,
                story_count=len(story_keys),
                stories=story_keys,
                epics=list(comp_epics),
            ))

        avg_conf = sum(k.confidence_score for k in knowledge_objects) / len(knowledge_objects) if knowledge_objects else 0
        avg_comp = sum(k.completeness_score for k in knowledge_objects) / len(knowledge_objects) if knowledge_objects else 0
        issues_with_ac = sum(1 for k in knowledge_objects if len(k.acceptance_criteria) > 0)
        issues_missing_priority = sum(1 for i in issues if i.priority == "Unspecified")
        issues_with_links = sum(1 for i in issues if len(i.issue_links) > 0)
        issues_missing_desc = sum(1 for i in issues if not i.description.strip())

        grade = "A" if avg_conf >= 0.8 else "B" if avg_conf >= 0.6 else "C" if avg_conf >= 0.4 else "D"

        project = ProjectKnowledge(
            project_key=issues[0].project_key if issues else "UNKNOWN",
            project_name=issues[0].project_name if issues else "Unknown",
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_issues=len(knowledge_objects),
            issues=knowledge_objects,
            epics=epic_summaries,
            components=comp_summaries,
            quality_metrics=QualityMetrics(
                total_issues=len(knowledge_objects),
                avg_confidence_score=round(avg_conf, 2),
                avg_completeness_score=round(avg_comp, 2),
                issues_with_ac=issues_with_ac,
                issues_missing_priority=issues_missing_priority,
                issues_with_links=issues_with_links,
                issues_missing_description=issues_missing_desc,
                quality_grade=grade,
            ),
            dependency_graph=_to_adjacency_list(self.state.relationships.get("dependency_graph", {})),
            metadata={
                "pipeline_version": "2.0.0",
                "extraction_llm": "deepseek/deepseek-chat",
                "consolidation_llm": "deepseek/deepseek-reasoner",
                "validation_llm": "deepseek/deepseek-chat",
                "stage_timings": json.dumps(self.state.stage_timings),
                "project_summary": consolidation.get("project_summary", consolidation.get("enrichment", {}).get("project_summary", "")),
            },
        )

        paths = publish_project_knowledge(project, str(output_dir))
        published.extend(paths.values())
        print("  ✅ Published project_knowledge.json + executive_summary.md")

        # 6. Validation report
        report_path = publish_validation_report(report, str(output_dir))
        published.append(report_path)
        print("  ✅ Published validation_report.md")

        self.state.published_files = published

        elapsed = time.time() - start_time
        self.state.stage_timings["publishing"] = round(elapsed, 2)
        print(f"  ⏱️  Completed in {elapsed:.2f}s")

        # Final summary
        print("\n" + "=" * 70)
        print("  🎉 PIPELINE COMPLETE (v2.0 — Refined Architecture)")
        print("=" * 70)
        print(f"  📁 Total files published: {len(published)}")
        print(f"  📊 Quality Grade: {grade} (Avg Confidence: {avg_conf:.2f})")
        print(f"  ⏱️  Total time: {sum(self.state.stage_timings.values()):.2f}s")
        print(f"  📂 Output directory: {output_dir.absolute()}")
        print()
        print("  Cache files (delete to re-run LLM stages):")
        print(f"    • {output_dir / 'knowledge_objects_cache.json'}")
        print(f"    • {output_dir / 'consolidation_cache.json'}")
        print(f"    • {output_dir / 'validation_results_cache.json'}")
        print(f"    • {output_dir / 'quality_report_cache.json'}")
        return "done"
