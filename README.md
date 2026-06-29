# JIRA Contextualization

Enterprise JIRA Contextualization Engine — From Raw Jira CSV Data to High-Quality, Validated & Reusable Structured Knowledge Base.

## Overview

This project implements a **6-stage offline pipeline** built with **CrewAI** that transforms raw Jira ticket exports (CSV) into a structured, validated knowledge base. It uses a hybrid approach combining deterministic rule-based processing with LLM-powered semantic extraction (DeepSeek Chat + Gemini 2.0 Flash).

## Architecture

Open `docs/architecture_diagram.html` in a browser to view the full pipeline architecture diagram.

## Pipeline Stages

| Stage | Name | LLM Used | Description |
|-------|------|----------|-------------|
| 1 | Ingestion & Normalization | ❌ No | Parse 1,646-column CSV, normalize into Pydantic models |
| 2 | Knowledge Extraction | ✅ DeepSeek Chat | Extract requirements, AC, business objectives using deterministic + LLM |
| 3 | Relationship & Hierarchy | ❌ No | Build epic hierarchy, dependency graphs, component groups |
| 4 | Knowledge Validation | ✅ Gemini 2.0 Flash | Validate completeness, detect ambiguity, score confidence |
| 5 | Knowledge Publisher | ❌ No | Publish 384 artifacts (JSON, Markdown, reports) |

## Quality Results

- **Quality Grade:** A
- **Avg Completeness:** 96%
- **Avg Confidence:** 90%
- **Critical Issues:** 0
- **Total Files Generated:** 384

## Project Structure

```
src/jira_contextualization/
├── config/
│   ├── agents.yaml          # AI agent definitions
│   └── tasks.yaml           # Task definitions
├── models/
│   ├── normalized_issue.py  # NormalizedIssue Pydantic model
│   ├── structured_knowledge.py  # StructuredIssueKnowledge model
│   ├── project_knowledge.py # ProjectKnowledge model
│   └── validation_report.py # QualityReport model
├── tools/
│   ├── csv_parser.py        # Index-based CSV parser
│   ├── wiki_markup_parser.py # Jira wiki markup parser
│   ├── requirement_extractor.py # Deterministic requirement extraction
│   ├── relationship_builder.py  # Epic/dependency graph builder
│   ├── knowledge_validator.py   # Completeness scoring
│   └── knowledge_publisher.py   # JSON + Markdown output
├── flow.py                  # CrewAI Flow orchestration
└── main.py                  # Entry point
```

## Setup

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Add your DEEPSEEK_API_KEY and GOOGLE_API_KEY

# Run the pipeline
uv run python -m jira_contextualization.main
```

## Tech Stack

- Python 3.12
- CrewAI Framework
- Pydantic v2
- DeepSeek Chat (extraction LLM)
- Gemini 2.0 Flash (validation LLM)
- UV package manager
