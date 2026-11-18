"""
Unified Document Parser & Ingestion Pipeline.
Routes files to specific loaders and produces indexed chunks.
"""

from pathlib import Path
from typing import List, Tuple
from app.ingestion.base_loader import BaseLoader, RawDocument, DocumentChunk
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.txt_loader import TxtLoader
from app.ingestion.tabular_loader import TabularLoader
from app.ingestion.chunker import DocumentChunker
from app.utils.logger import logger


class DocumentParser:
    """Universal parser supporting PDF, DOCX, TXT, CSV, and Excel."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._loaders: dict[str, BaseLoader] = {
            ".pdf": PDFLoader(),
            ".docx": DocxLoader(),
            ".txt": TxtLoader(),
            ".md": TxtLoader(),
            ".csv": TabularLoader(),
            ".xlsx": TabularLoader(),
            ".xls": TabularLoader(),
        }

    def parse_file(self, file_path: str | Path) -> Tuple[RawDocument, List[DocumentChunk]]:
        """Parses a file and returns the RawDocument along with its DocumentChunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        loader = self._loaders.get(ext)
        if not loader:
            raise ValueError(
                f"Unsupported file format '{ext}'. Supported formats: {list(self._loaders.keys())}"
            )

        raw_doc = loader.load(path)
        chunks = self.chunker.chunk_document(raw_doc)
        logger.info(f"Successfully processed {path.name}: {len(chunks)} chunks created.")
        return raw_doc, chunks
