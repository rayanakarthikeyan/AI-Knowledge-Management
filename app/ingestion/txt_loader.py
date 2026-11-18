"""
Plain text and Markdown loader with multiple encoding detection.
"""

from pathlib import Path
from app.ingestion.base_loader import BaseLoader, RawDocument
from app.utils.logger import logger


class TxtLoader(BaseLoader):
    """Extracts text from .txt and markdown files with fallback encodings."""

    def load(self, file_path: Path) -> RawDocument:
        logger.info(f"Loading text document: {file_path.name}")
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        content = None
        used_encoding = "utf-8"

        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                used_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise ValueError(f"Unable to decode text file '{file_path.name}' with supported encodings.")

        # Split into rough paragraphs/sections
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        sections = [
            {"section_index": idx, "text": p}
            for idx, p in enumerate(paragraphs, start=1)
        ]

        return RawDocument(
            content=content,
            source_filename=file_path.name,
            file_type="txt",
            pages_or_sections=sections,
            metadata={
                "encoding": used_encoding,
                "paragraphs_count": len(paragraphs),
                "file_size_bytes": file_path.stat().st_size
            }
        )
