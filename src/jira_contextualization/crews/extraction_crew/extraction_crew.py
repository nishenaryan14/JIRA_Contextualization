"""Extraction Crew — Extracts structured knowledge from a single Jira issue.

Uses 2 specialized agents in sequential process:
  1. RequirementExtractor: Core requirements, AC, business rules
  2. DiscussionAnalyzer: Risks, decisions, open questions
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class ExtractionCrew:
    """Crew for extracting structured knowledge from a single Jira ticket."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        super().__init__()
        self._llm = LLM(
            model="deepseek/deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.1,
            max_tokens=8000,
        )

    @agent
    def requirement_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["requirement_extractor"],  # type: ignore[index]
            llm=self._llm,
            verbose=False,
            max_iter=3,
        )

    @agent
    def discussion_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["discussion_analyzer"],  # type: ignore[index]
            llm=self._llm,
            verbose=False,
            max_iter=3,
        )

    @task
    def extract_requirements(self) -> Task:
        return Task(
            config=self.tasks_config["extract_requirements"],  # type: ignore[index]
        )

    @task
    def analyze_discussions(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_discussions"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Extraction Crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )
