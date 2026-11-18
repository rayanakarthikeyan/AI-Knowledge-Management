"""
PDF Document Loader using pypdf.
Extracts page-by-page text with page number attribution.
"""

from pathlib import Path
from pypdf import PdfReader
from app.ingestion.base_loader import BaseLoader, RawDocument
from app.utils.logger import logger


class PDFLoader(BaseLoader):
    """Extracts text and page metadata from PDF files."""

    def load(self, file_path: Path) -> RawDocument:
        logger.info(f"Loading PDF document: {file_path.name}")
        try:
            reader = PdfReader(str(file_path))
            pages_data = []
            full_text_parts = []

            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_text_clean = page_text.strip()
                if page_text_clean:
                    pages_data.append({
                        "page_number": idx,
                        "text": page_text_clean
                    })
                    full_text_parts.append(f"[Page {idx}]\n{page_text_clean}")

            full_content = "\n\n".join(full_text_parts)

            return RawDocument(
                content=full_content,
                source_filename=file_path.name,
                file_type="pdf",
                pages_or_sections=pages_data,
                metadata={
                    "total_pages": len(reader.pages),
                    "file_size_bytes": file_path.stat().st_size
                }
            )
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path.name}: {e}")
            raise ValueError(f"Could not read PDF file '{file_path.name}': {str(e)}")
