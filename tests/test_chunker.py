"""
Unit tests for DocumentChunker.
"""

from app.ingestion.base_loader import RawDocument
from app.ingestion.chunker import DocumentChunker


def test_chunk_continuous_text():
    text = "Sentence one. " * 50
    raw_doc = RawDocument(
        content=text,
        source_filename="test.txt",
        file_type="txt",
        metadata={"author": "Enterprise Admin"}
    )

    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document(raw_doc)

    assert len(chunks) > 1
    for ch in chunks:
        assert ch.source_filename == "test.txt"
        assert ch.metadata["author"] == "Enterprise Admin"
        assert len(ch.content) <= 200


def test_chunk_tabular_records():
    sections = [
        {"sheet_name": "Sales", "row_index": 1, "text": "Product: Widget A | Units: 100"},
        {"sheet_name": "Sales", "row_index": 2, "text": "Product: Widget B | Units: 250"},
        {"sheet_name": "Sales", "row_index": 3, "text": "Product: Widget C | Units: 300"},
    ]
    raw_doc = RawDocument(
        content="...",
        source_filename="sales.xlsx",
        file_type="tabular",
        pages_or_sections=sections,
        metadata={"total_rows": 3}
    )

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(raw_doc)

    assert len(chunks) >= 1
    assert any("Widget A" in c.content for c in chunks)
    assert chunks[0].file_type == "tabular"
