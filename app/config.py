"""
Enterprise AI Knowledge Management Configuration
Handles application settings, model configurations, and safety thresholds.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


class AppConfig(BaseModel):
    """Central configuration class for the enterprise application."""

    # Application Metadata
    APP_NAME: str = "Enterprise AI Knowledge Hub"
    APP_VERSION: str = "1.0.0"

    # Supported Document Types
    SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"]

    # Ingestion & Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Vector Database
    CHROMA_PERSIST_DIRECTORY: str = str(VECTOR_STORE_DIR)
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "enterprise_kb")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Retrieval
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    DEFAULT_GEMINI_MODEL: str = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-1.5-flash")
    DEFAULT_OPENAI_MODEL: str = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
    DEFAULT_GROQ_MODEL: str = os.getenv("DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Multi-Agent Flags
    ENABLE_HALLUCINATION_VALIDATION: bool = os.getenv("ENABLE_HALLUCINATION_VALIDATION", "true").lower() == "true"
    ENABLE_SAFETY_GUARDRAILS: bool = os.getenv("ENABLE_SAFETY_GUARDRAILS", "true").lower() == "true"
    MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "3"))


config = AppConfig()
