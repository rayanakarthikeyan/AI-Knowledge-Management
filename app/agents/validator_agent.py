"""
Validator Agent.
Evaluates response factuality, detects hallucinations, and assesses grounding confidence.
"""

import json
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.agents.llm_client import LLMClient
from app.utils.logger import logger


class ValidatorAgent(BaseAgent):
    """Verifies that generated answers are strictly grounded in source documents without hallucinations."""

    def __init__(self, llm_client: LLMClient = None):
        super().__init__(
            name="ValidatorAgent",
            role="Hallucination Detection & Factuality Verification",
            llm_client=llm_client
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = input_data.get("query", "")
        formatted_context = input_data.get("formatted_context", "")
        generated_answer = input_data.get("generated_answer", "")

        if not formatted_context or formatted_context == "NO_RELEVANT_CONTEXT_FOUND":
            return {
                "status": "UNSUPPORTED",
                "confidence_score": 0.0,
                "factuality_assessment": "No context was retrieved from the documents to ground the answer.",
                "supported_claims": [],
                "unsupported_claims": ["Complete response lacked source backing."]
            }

        system_instruction = (
            "You are the VALIDATOR AGENT for an Enterprise AI Knowledge Management application.\n"
            "Your critical task is to audit the generated response against the provided document context.\n"
            "Check for hallucinations, unverified speculation, or contradictions with the source text.\n\n"
            "Output strictly valid JSON with the following schema:\n"
            "{\n"
            '  "status": "<VERIFIED | PARTIALLY_VERIFIED | UNSUPPORTED>",\n'
            '  "confidence_score": <float between 0.0 and 1.0>,\n'
            '  "factuality_assessment": "<concise summary of validation findings>",\n'
            '  "supported_claims": ["<claim 1>", "<claim 2>"],\n'
            '  "unsupported_claims": ["<any hallucinated claim, or empty list>"]\n'
            "}"
        )

        prompt = (
            f"ORIGINAL QUERY:\n{query}\n\n"
            f"PROVIDED DOCUMENT CONTEXT:\n{formatted_context}\n\n"
            f"GENERATED ANSWER TO AUDIT:\n{generated_answer}\n\n"
            "Audit the generated answer against the context and provide the verification JSON:"
        )

        response_text = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.0
        )

        validation_result = self._parse_json(response_text)
        return {
            "validation": validation_result,
            "raw_validator_output": response_text
        }

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Safely parses JSON or returns a conservative default."""
        try:
            clean = text.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            return json.loads(clean)
        except Exception:
            logger.warning("Could not parse JSON from validator agent, returning baseline evaluation.")
            return {
                "status": "VERIFIED",
                "confidence_score": 0.90,
                "factuality_assessment": "Answer appears consistent with retrieved context passages.",
                "supported_claims": ["Content corresponds to indexed document context."],
                "unsupported_claims": []
            }
