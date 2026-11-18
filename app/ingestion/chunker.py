"""
Text and Tabular Chunker.
Splits document content into overlapping semantic chunks with full provenance metadata.
"""

import uuid
from typing import List
from app.ingestion.base_loader import RawDocument, DocumentChunk
from app.config import config
from app.utils.logger import logger


class DocumentChunker:
    """Chunks documents into semantic segments with provenance metadata."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    def chunk_document(self, raw_doc: RawDocument) -> List[DocumentChunk]:
        """Processes a RawDocument into a list of DocumentChunks."""
        chunks: List[DocumentChunk] = []

        if raw_doc.file_type == "tabular":
            # For tabular data, chunk by rows/records or grouped rows
            chunks = self._chunk_tabular(raw_doc)
        elif raw_doc.pages_or_sections and raw_doc.file_type == "pdf":
            # For PDF, chunk by pages or page sections
            chunks = self._chunk_pdf_pages(raw_doc)
        else:
            # For DOCX, TXT, or continuous text
            chunks = self._chunk_continuous_text(raw_doc)

        logger.info(f"Generated {len(chunks)} chunks for {raw_doc.source_filename}")
        return chunks

    def _chunk_continuous_text(self, raw_doc: RawDocument) -> List[DocumentChunk]:
        """Splits continuous text using sliding window with character or word boundaries."""
        text = raw_doc.content
        chunks = []
        start = 0
        chunk_idx = 0
        total_len = len(text)

        if total_len == 0:
            return []

        while start < total_len:
            end = min(start + self.chunk_size, total_len)

            # Try to snap to the nearest newline or space to prevent word slicing
            if end < total_len:
                last_newline = text.rfind("\n", start + self.chunk_size // 2, end)
                if last_newline != -1:
                    end = last_newline + 1
                else:
                    last_space = text.rfind(" ", start + self.chunk_size // 2, end)
                    if last_space != -1:
                        end = last_space + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{raw_doc.source_filename}_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                meta = {
                    "source": raw_doc.source_filename,
                    "file_type": raw_doc.file_type,
                    "char_start": start,
                    "char_end": end,
                    **raw_doc.metadata
                }
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source_filename=raw_doc.source_filename,
                    file_type=raw_doc.file_type,
                    chunk_index=chunk_idx,
                    metadata=meta,
                    chunk_id=chunk_id
                ))
                chunk_idx += 1

            if end >= total_len:
                break

            # Slide start forward
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _chunk_pdf_pages(self, raw_doc: RawDocument) -> List[DocumentChunk]:
        """Chunks PDF pages, retaining exact page numbers in metadata."""
        chunks = []
        chunk_idx = 0

        for page_data in raw_doc.pages_or_sections:
            page_num = page_data.get("page_number", 1)
            page_text = page_data.get("text", "").strip()

            if not page_text:
                continue

            # If page text is within chunk size, keep it as one chunk
            if len(page_text) <= self.chunk_size:
                chunk_id = f"{raw_doc.source_filename}_p{page_num}_{chunk_idx}"
                chunks.append(DocumentChunk(
                    content=page_text,
                    source_filename=raw_doc.source_filename,
                    file_type="pdf",
                    chunk_index=chunk_idx,
                    metadata={
                        "source": raw_doc.source_filename,
                        "file_type": "pdf",
                        "page_number": page_num,
                        **raw_doc.metadata
                    },
                    chunk_id=chunk_id
                ))
                chunk_idx += 1
            else:
                # Sub-chunk page
                sub_start = 0
                while sub_start < len(page_text):
                    sub_end = min(sub_start + self.chunk_size, len(page_text))
                    sub_text = page_text[sub_start:sub_end].strip()
                    if sub_text:
                        chunk_id = f"{raw_doc.source_filename}_p{page_num}_{chunk_idx}"
                        chunks.append(DocumentChunk(
                            content=sub_text,
                            source_filename=raw_doc.source_filename,
                            file_type="pdf",
                            chunk_index=chunk_idx,
                            metadata={
                                "source": raw_doc.source_filename,
                                "file_type": "pdf",
                                "page_number": page_num,
                                **raw_doc.metadata
                            },
                            chunk_id=chunk_id
                        ))
                        chunk_idx += 1
                    if sub_end >= len(page_text):
                        break
                    sub_start = max(sub_start + 1, sub_end - self.chunk_overlap)

        return chunks

    def _chunk_tabular(self, raw_doc: RawDocument) -> List[DocumentChunk]:
        """Groups tabular rows into chunks to preserve row relationships."""
        chunks = []
        chunk_idx = 0
        current_batch = []
        current_len = 0
        current_sheet = None

        for sec in raw_doc.pages_or_sections:
            text = sec.get("text", "")
            sheet = sec.get("sheet_name", "Sheet1")
            row_idx = sec.get("row_index", 0)

            if current_sheet is None:
                current_sheet = sheet

            # If sheet changes or batch exceeds size, flush batch
            if (sheet != current_sheet) or (current_len + len(text) > self.chunk_size and current_batch):
                batch_text = "\n".join(current_batch)
                chunk_id = f"{raw_doc.source_filename}_{current_sheet}_{chunk_idx}"
                chunks.append(DocumentChunk(
                    content=batch_text,
                    source_filename=raw_doc.source_filename,
                    file_type="tabular",
                    chunk_index=chunk_idx,
                    metadata={
                        "source": raw_doc.source_filename,
                        "file_type": raw_doc.file_type,
                        "sheet_name": current_sheet,
                        "records_in_chunk": len(current_batch),
                        **raw_doc.metadata
                    },
                    chunk_id=chunk_id
                ))
                chunk_idx += 1
                current_batch = []
                current_len = 0
                current_sheet = sheet

            current_batch.append(text)
            current_len += len(text)

        # Flush remaining
        if current_batch:
            batch_text = "\n".join(current_batch)
            chunk_id = f"{raw_doc.source_filename}_{current_sheet}_{chunk_idx}"
            chunks.append(DocumentChunk(
                content=batch_text,
                source_filename=raw_doc.source_filename,
                file_type="tabular",
                chunk_index=chunk_idx,
                metadata={
                    "source": raw_doc.source_filename,
                    "file_type": raw_doc.file_type,
                    "sheet_name": current_sheet or "Sheet1",
                    "records_in_chunk": len(current_batch),
                    **raw_doc.metadata
                },
                chunk_id=chunk_id
            ))

        return chunks
