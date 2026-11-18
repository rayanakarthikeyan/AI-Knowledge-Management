"""
Planner Agent.
Decomposes user queries into actionable retrieval tasks, queries, and constraints.
"""

import json
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.agents.llm_client import LLMClient
from app.utils.logger import logger


class PlannerAgent(BaseAgent):
    """Analyzes user inquiry and establishes high-precision execution strategy."""

    def __init__(self, llm_client: LLMClient = None):
        super().__init__(
            name="PlannerAgent",
            role="Query Analysis & Strategic Task Planning",
            llm_client=llm_client
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        available_docs = input_data.get("available_docs", [])

        docs_summary = ", ".join(d.get("filename", "") for d in available_docs) or "None uploaded yet"

        system_instruction = (
            "You are the PLANNER AGENT for an Enterprise Knowledge Management system. "
            "Analyze the user's natural language inquiry and establish an optimal retrieval and reasoning plan. "
            "Identify the user's intent (e.g., Factual Retrieval, Cross-document Synthesis, Tabular Analysis, Aggregation), "
            "generate 2 to 3 targeted semantic and keyword search queries, determine if a specific file or sheet is referenced, "
            "and decompose the inquiry into clear subtasks.\n\n"
            "Output strictly valid JSON with this exact schema:\n"
            "{\n"
            '  "intent": "<string>",\n'
            '  "search_queries": ["<dense query>", "<lexical query>"],\n'
            '  "document_filter": "<filename or null>",\n'
            '  "subtasks": ["<subtask 1>", "<subtask 2>"]\n'
            "}"
        )

        prompt = (
            f"User Question: {query}\n"
            f"Available Documents: {docs_summary}\n\n"
            "Formulate the execution plan in JSON:"
        )

        response_text = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1
        )

        plan = self._parse_json(response_text, query)
        return {
            "plan": plan,
            "raw_planner_output": response_text
        }

    def _parse_json(self, text: str, fallback_query: str) -> Dict[str, Any]:
        """Safely parses JSON or provides structured fallback."""
        try:
            clean = text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            return json.loads(clean)
        except Exception:
            logger.warning("Could not parse JSON from planner agent; applying structured heuristic.")
            return {
                "intent": "Information Retrieval",
                "search_queries": [fallback_query],
                "document_filter": None,
                "subtasks": [
                    f"Retrieve relevant passages for '{fallback_query}'",
                    "Synthesize findings with source citations",
                    "Verify factual accuracy against source context"
                ]
            }
