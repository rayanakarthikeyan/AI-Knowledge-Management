"""
Prompt Injection Guard and PII Detection.
Protects the enterprise knowledge base against jailbreaks and data leakage.
"""

import re
from typing import Tuple, List
from app.utils.logger import logger


class PromptGuard:
    """Detects adversarial inputs, prompt injection attempts, and sensitive data."""

    # Common injection and system bypass patterns
    INJECTION_PATTERNS = [
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        r"(?i)\bdisregard\s+(all\s+)?(previous|prior|system)\b",
        r"(?i)\byou\s+are\s+now\s+(an\s+unfiltered|in\s+developer\s+mode|dan)\b",
        r"(?i)\bsystem\s+prompt\s*(reveal|leak|show|print|display)\b",
        r"(?i)\brepeat\s+(everything|the\s+prompt|all\s+text)\s+above\b",
        r"(?i)\bbypass\s+(safety|content|security)\s+filters\b",
        r"(?i)\bdo\s+anything\s+now\b",
        r"(?i)\bpretend\s+you\s+have\s+no\s+(rules|restrictions|guidelines)\b",
    ]

    # Regex patterns for sensitive PII
    PII_PATTERNS = {
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "API_KEY": r"\b(?:sk-[a-zA-Z0-9]{32,}|ghp_[a-zA-Z0-9]{36}|AIza[0-9A-Za-z-_]{35})\b",
    }

    def __init__(self, max_query_length: int = 1500):
        self.max_query_length = max_query_length
        self._compiled_injections = [re.compile(p) for p in self.INJECTION_PATTERNS]

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validates query for safety and prompt injection.
        Returns: (is_safe: bool, reason_or_message: str)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty."

        if len(query) > self.max_query_length:
            return False, f"Query exceeds maximum character limit of {self.max_query_length}."

        # Check injection patterns
        for pattern in self._compiled_injections:
            if pattern.search(query):
                logger.warning(f"Prompt injection pattern detected in query: {query[:80]}...")
                return False, "Security Notice: The input contains prohibited system override or prompt injection phrases."

        return True, "Safe"

    def mask_pii(self, text: str) -> Tuple[str, List[str]]:
        """
        Detects and masks sensitive PII in text or outputs.
        Returns: (masked_text, list_of_detected_types)
        """
        detected = []
        masked_text = text
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, masked_text)
            if matches:
                detected.append(pii_type)
                masked_text = re.sub(pattern, f"[REDACTED_{pii_type}]", masked_text)

        return masked_text, detected
