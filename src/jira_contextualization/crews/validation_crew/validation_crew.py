"""Validation Crew — Validates knowledge quality for a single issue.

Uses DeepSeek Chat for validation to avoid Gemini quota limits.
"""

from __future__ import annotations

import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class ValidationCrew:
    """Crew for validating knowledge quality per issue."""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        super().__init__()
        # Use DeepSeek Chat instead of Gemini to avoid quota limits
        self._llm = LLM(
            model="deepseek/deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.2,
            max_tokens=8000,
        )

    @agent
    def validation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["validation_agent"],  # type: ignore[index]
            llm=self._llm,
            verbose=False,
            max_iter=3,
        )

    @task
    def validate_knowledge(self) -> Task:
        return Task(
            config=self.tasks_config["validate_knowledge"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )
