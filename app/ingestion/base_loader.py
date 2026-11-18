"""
Base Document and Loader definitions for multi-format ingestion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    """Represents a discrete text chunk with full citation metadata."""
    content: str
    source_filename: str
    file_type: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_filename": self.source_filename,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }


@dataclass
class RawDocument:
    """Represents raw extracted content from a document before chunking."""
    content: str
    source_filename: str
    file_type: str
    pages_or_sections: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract base class for format-specific document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> RawDocument:
        """Parses the given file into a RawDocument instance."""
        pass
