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
    """Analyzes the user's inquiry and establishes an execution strategy."""

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
            "Your objective is to analyze the user's natural language question and formulate an optimal plan. "
            "Identify the user's intent (e.g. Factual Lookup, Cross-document Comparison, Tabular Calculation, Summarization), "
            "generate specific semantic search queries, determine if a specific file filter is mentioned, "
            "and outline sub-steps.\n\n"
            "Output strictly valid JSON with this format:\n"
            "{\n"
            '  "intent": "<string>",\n'
            '  "search_queries": ["<query 1>", "<query 2>"],\n'
            '  "document_filter": "<filename or null>",\n'
            '  "subtasks": ["<step 1>", "<step 2>"]\n'
            "}"
        )

        prompt = (
            f"User Question: {query}\n"
            f"Available Documents in Knowledge Base: {docs_summary}\n\n"
            "Provide the task plan in JSON format:"
        )

        response_text = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1
        )

        # Parse JSON output
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
            logger.warning("Could not parse JSON from planner agent, applying structured heuristic.")
            return {
                "intent": "Information Retrieval",
                "search_queries": [fallback_query],
                "document_filter": None,
                "subtasks": [
                    f"Retrieve passages for '{fallback_query}'",
                    "Synthesize information into accurate answer",
                    "Validate answer against source documents"
                ]
            }
