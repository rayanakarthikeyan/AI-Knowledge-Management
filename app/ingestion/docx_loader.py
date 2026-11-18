"""
Word Document (DOCX) Loader using python-docx.
Extracts paragraphs, headings, and tables.
"""

from pathlib import Path
from docx import Document
from app.ingestion.base_loader import BaseLoader, RawDocument
from app.utils.logger import logger


class DocxLoader(BaseLoader):
    """Extracts text, headings, and table cells from DOCX files."""

    def load(self, file_path: Path) -> RawDocument:
        logger.info(f"Loading DOCX document: {file_path.name}")
        try:
            doc = Document(str(file_path))
            sections_data = []
            full_text_parts = []

            # Extract paragraphs
            for idx, p in enumerate(doc.paragraphs, start=1):
                text = p.text.strip()
                if text:
                    is_heading = p.style.name.startswith("Heading") if p.style else False
                    sections_data.append({
                        "section_index": idx,
                        "is_heading": is_heading,
                        "text": text
                    })
                    prefix = f"### {text}" if is_heading else text
                    full_text_parts.append(prefix)

            # Extract tables if any
            for t_idx, table in enumerate(doc.tables, start=1):
                table_lines = [f"[Table {t_idx}]"]
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells]
                    table_lines.append(" | ".join(row_cells))
                if len(table_lines) > 1:
                    table_text = "\n".join(table_lines)
                    full_text_parts.append(table_text)
                    sections_data.append({
                        "section_index": len(sections_data) + 1,
                        "is_table": True,
                        "text": table_text
                    })

            full_content = "\n\n".join(full_text_parts)

            return RawDocument(
                content=full_content,
                source_filename=file_path.name,
                file_type="docx",
                pages_or_sections=sections_data,
                metadata={
                    "paragraphs_count": len(doc.paragraphs),
                    "tables_count": len(doc.tables),
                    "file_size_bytes": file_path.stat().st_size
                }
            )
        except Exception as e:
            logger.error(f"Failed to parse DOCX {file_path.name}: {e}")
            raise ValueError(f"Could not read DOCX file '{file_path.name}': {str(e)}")
