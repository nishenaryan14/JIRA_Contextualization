"""Consolidation Crew — Cross-issue knowledge consolidation and enrichment.

Processes ALL issues together (not per-issue) to:
  1. Detect duplicate requirements across issues
  2. Generate project-level enrichment (summaries, domains, themes)

Uses DeepSeek Reasoner for deeper cross-issue reasoning.
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class ConsolidationCrew:
    """Crew for cross-issue consolidation and project-level enrichment."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        super().__init__()
        # Use DeepSeek Reasoner for deeper cross-issue reasoning
        self._llm = LLM(
            model="deepseek/deepseek-reasoner",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.0,
            max_tokens=16000,
        )

    @agent
    def consolidation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["consolidation_agent"],  # type: ignore[index]
            llm=self._llm,
            verbose=False,
            max_iter=5,
        )

    @task
    def detect_duplicates(self) -> Task:
        return Task(
            config=self.tasks_config["detect_duplicates"],  # type: ignore[index]
        )

    @task
    def enrich_project_knowledge(self) -> Task:
        return Task(
            config=self.tasks_config["enrich_project_knowledge"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )
