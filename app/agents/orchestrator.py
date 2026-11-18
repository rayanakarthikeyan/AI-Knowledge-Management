"""
Multi-Agent Orchestrator.
Coordinates the entire lifecycle: Safety -> Planner -> Retriever -> Reasoning -> Validator.
"""

from typing import Any, Dict, List
from app.config import config
from app.safety.prompt_guard import PromptGuard
from app.vectorstore.chroma_store import ChromaVectorStore
from app.agents.llm_client import LLMClient
from app.agents.planner_agent import PlannerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.validator_agent import ValidatorAgent
from app.utils.logger import logger


class AgentOrchestrator:
    """Manages multi-agent execution pipeline with telemetry and safety guardrails."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        llm_client: LLMClient = None,
        prompt_guard: PromptGuard = None,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client or LLMClient()
        self.prompt_guard = prompt_guard or PromptGuard()

        # Instantiate specialized agents
        self.planner = PlannerAgent(llm_client=self.llm_client)
        self.retriever = RetrieverAgent(vector_store=self.vector_store)
        self.reasoning = ReasoningAgent(llm_client=self.llm_client)
        self.validator = ValidatorAgent(llm_client=self.llm_client)

    def process_query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Runs the complete multi-agent workflow for a user inquiry."""
        workflow_trace: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # Step 0: Safety & Security Guardrail Check
        # -------------------------------------------------------------
        if config.ENABLE_SAFETY_GUARDRAILS:
            is_safe, safety_msg = self.prompt_guard.validate_query(query)
            if not is_safe:
                return {
                    "answer": f"[SECURITY GUARDRAIL TRIGGERED] Request Blocked: {safety_msg}",
                    "status": "blocked_by_safety",
                    "citations": [],
                    "validation": {
                        "status": "BLOCKED",
                        "confidence_score": 0.0,
                        "factuality_assessment": safety_msg
                    },
                    "workflow_trace": [{
                        "agent": "SafetyGuard",
                        "role": "Input Validation",
                        "status": "violation_detected",
                        "details": safety_msg
                    }]
                }

        # -------------------------------------------------------------
        # Step 1: Planner Agent
        # -------------------------------------------------------------
        available_docs = self.vector_store.list_documents()
        planner_input = {"query": query, "available_docs": available_docs}
        planner_output = self.planner.execute_with_telemetry(planner_input)
        workflow_trace.append(planner_output.get("_telemetry", {}))

        plan = planner_output.get("plan", {})

        # -------------------------------------------------------------
        # Step 2: Retriever Agent (Hybrid Search)
        # -------------------------------------------------------------
        retriever_input = {
            "query": query,
            "plan": plan,
            "top_k": top_k or config.TOP_K_RETRIEVAL
        }
        retriever_output = self.retriever.execute_with_telemetry(retriever_input)
        workflow_trace.append(retriever_output.get("_telemetry", {}))

        formatted_context = retriever_output.get("formatted_context", "")
        citations = retriever_output.get("citations", [])

        # -------------------------------------------------------------
        # Step 3: Reasoning Agent
        # -------------------------------------------------------------
        reasoning_input = {
            "query": query,
            "formatted_context": formatted_context,
            "plan": plan
        }
        reasoning_output = self.reasoning.execute_with_telemetry(reasoning_input)
        workflow_trace.append(reasoning_output.get("_telemetry", {}))

        raw_answer = reasoning_output.get("response", "")

        # -------------------------------------------------------------
        # Step 4: Validator Agent (Factuality Audit)
        # -------------------------------------------------------------
        validation_info = {}
        if config.ENABLE_HALLUCINATION_VALIDATION and retriever_output.get("retrieved_count", 0) > 0:
            validator_input = {
                "query": query,
                "formatted_context": formatted_context,
                "generated_answer": raw_answer
            }
            validator_output = self.validator.execute_with_telemetry(validator_input)
            workflow_trace.append(validator_output.get("_telemetry", {}))
            validation_info = validator_output.get("validation", {})
        else:
            validation_info = {
                "status": "UNSUPPORTED" if retriever_output.get("retrieved_count", 0) == 0 else "SKIPPED",
                "confidence_score": 0.0 if retriever_output.get("retrieved_count", 0) == 0 else 0.85,
                "factuality_assessment": "Zero matching passages retrieved from knowledge base."
            }

        # Mask sensitive PII
        final_answer, detected_pii = self.prompt_guard.mask_pii(raw_answer)

        return {
            "answer": final_answer,
            "plan": plan,
            "citations": citations,
            "validation": validation_info,
            "workflow_trace": workflow_trace,
            "retrieved_count": retriever_output.get("retrieved_count", 0),
            "detected_pii": detected_pii
        }
