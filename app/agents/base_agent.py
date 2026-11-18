"""
Base Agent Interface.
Standardizes agent execution lifecycle, telemetry, and structured outputs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import time
from app.agents.llm_client import LLMClient
from app.utils.logger import logger


class BaseAgent(ABC):
    """Abstract class for all specialized AI agents."""

    def __init__(self, name: str, role: str, llm_client: LLMClient = None):
        self.name = name
        self.role = role
        self.llm_client = llm_client or LLMClient()

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the agent's core capability."""
        pass

    def execute_with_telemetry(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Wraps execution with duration tracking and structured logging."""
        start_time = time.time()
        logger.info(f"[{self.name}] Initiating step: {self.role}")
        try:
            result = self.run(input_data)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            result["_telemetry"] = {
                "agent": self.name,
                "role": self.role,
                "duration_ms": duration_ms,
                "status": "success"
            }
            logger.info(f"[{self.name}] Completed successfully in {duration_ms}ms")
            return result
        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(f"[{self.name}] Execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "_telemetry": {
                    "agent": self.name,
                    "role": self.role,
                    "duration_ms": duration_ms,
                    "status": "error"
                }
            }
