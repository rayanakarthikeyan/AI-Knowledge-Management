"""
Reasoning Agent.
Synthesizes retrieved context into an accurate, grounded, and well-structured answer with citations.
"""

from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.agents.llm_client import LLMClient
from app.utils.logger import logger


class ReasoningAgent(BaseAgent):
    """Performs contextual inference and synthesis over retrieved enterprise knowledge."""

    def __init__(self, llm_client: LLMClient = None):
        super().__init__(
            name="ReasoningAgent",
            role="Contextual Synthesis & Response Formulation",
            llm_client=llm_client
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        formatted_context = input_data.get("formatted_context", "")
        plan = input_data.get("plan", {})
        intent = plan.get("intent", "Information Retrieval")

        if not formatted_context or formatted_context == "NO_RELEVANT_CONTEXT_FOUND":
            return {
                "response": (
                    "No relevant information was found in the indexed enterprise documents "
                    "to answer this inquiry. Please verify that the pertinent document "
                    "(PDF, Word, TXT, CSV, or Excel) has been ingested into the repository."
                ),
                "has_context": False
            }

        system_instruction = (
            "You are the REASONING AGENT of an Enterprise Knowledge Management application.\n"
            "Your objective is to provide a comprehensive, accurate, and professional response "
            "strictly grounded in the provided document context.\n\n"
            "Directives:\n"
            "1. Ground all claims directly in the facts, metrics, and excerpts given in the CONTEXT blocks.\n"
            "2. Cite every factual assertion using brackets corresponding to the source index (e.g., [1], [2]).\n"
            "3. Maintain a formal, executive tone without informal vernacular or emojis.\n"
            "4. For CSV or Excel tabular data, provide exact quantities, currencies, and specify row or sheet names.\n"
            "5. If a specific portion of the inquiry is not addressed in the context, explicitly indicate the omission.\n"
            "6. Use structured formatting (bullet points, clear paragraphs, comparison tables) to maximize clarity."
        )

        prompt = (
            f"USER INQUIRY:\n{query}\n\n"
            f"DETECTED INTENT: {intent}\n\n"
            f"DOCUMENT CONTEXT:\n{formatted_context}\n\n"
            "Provide a grounded, professional response with citations:"
        )

        answer = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.15
        )

        return {
            "response": answer,
            "has_context": True
        }
