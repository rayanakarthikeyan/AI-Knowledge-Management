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
            role="Contextual Synthesis & Response Generation",
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
                    "I was unable to find relevant information in the uploaded enterprise documents "
                    "to answer your question. Please ensure the relevant files (PDF, DOCX, TXT, CSV, or Excel) "
                    "are uploaded to the knowledge base."
                ),
                "has_context": False
            }

        system_instruction = (
            "You are the REASONING AGENT of an Enterprise Knowledge Management system.\n"
            "Your task is to answer the user's question accurately, strictly grounded in the provided document context.\n"
            "Rules:\n"
            "1. Rely strictly on the facts, figures, and data provided in the CONTEXT blocks.\n"
            "2. Whenever making a specific claim or presenting data, cite the source using the source index (e.g., [1], [2]).\n"
            "3. If the context does not contain enough information to fully answer a part of the question, clearly state what is missing.\n"
            "4. For CSV or Excel tabular data, provide exact numbers and clarify sheet/row details where relevant.\n"
            "5. Structure your response clearly with headings or bullet points where appropriate."
        )

        prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"REPRESENTED INTENT: {intent}\n\n"
            f"DOCUMENT CONTEXT:\n{formatted_context}\n\n"
            "Provide a comprehensive, accurate, and cited answer:"
        )

        answer = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.2
        )

        return {
            "response": answer,
            "has_context": True
        }
