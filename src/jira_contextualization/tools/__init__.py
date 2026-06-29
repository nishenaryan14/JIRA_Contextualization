"""Deterministic tool modules for the Jira Contextualization pipeline.

This package contains standalone Python functions and classes that perform
the *deterministic* (non-LLM) stages of the pipeline.  Each module can be
used independently or composed into CrewAI tool wrappers at a later stage.

Modules
-------
- :mod:`csv_parser`           — Parse HP Jira CSV exports → ``NormalizedIssue``.
- :mod:`wiki_markup_parser`   — Parse Jira wiki markup into structured dicts.
- :mod:`requirement_extractor`— Regex-based extraction of AC, GWT, user stories.
- :mod:`relationship_builder` — Epic hierarchies, dependency graphs, clusters.
- :mod:`knowledge_validator`  — Quality checks and completeness scoring.
- :mod:`knowledge_publisher`  — Serialise knowledge to JSON / Markdown files.
"""

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
    clean_markup,
    extract_sections,
    parse_jira_markup,
)

__all__ = [
    # csv_parser
    "parse_jira_csv",
    # wiki_markup_parser
    "parse_jira_markup",
    "extract_sections",
    "clean_markup",
    # requirement_extractor
    "extract_requirements_deterministic",
    # relationship_builder
    "build_epic_hierarchy",
    "build_dependency_graph",
    "group_by_component",
    "find_related_clusters",
    # knowledge_validator
    "validate_issue",
    "calculate_completeness_score",
    "generate_quality_report",
    # knowledge_publisher
    "publish_issue_json",
    "publish_issue_markdown",
    "publish_project_knowledge",
    "publish_validation_report",
]
