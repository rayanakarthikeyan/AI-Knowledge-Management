"""
Unit tests for ChromaVectorStore.
"""

from pathlib import Path
from app.ingestion.base_loader import DocumentChunk
from app.vectorstore.chroma_store import ChromaVectorStore


def test_chroma_add_and_search(tmp_path: Path):
    store = ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_collection"
    )

    chunks = [
        DocumentChunk(
            content="Enterprise compliance guidelines require annual employee background checks.",
            source_filename="compliance.pdf",
            file_type="pdf",
            chunk_index=0,
            metadata={"department": "Legal", "page_number": 3}
        ),
        DocumentChunk(
            content="Q3 sales grew by 25% year-over-year in the Asia-Pacific regional sector.",
            source_filename="q3_report.docx",
            file_type="docx",
            chunk_index=0,
            metadata={"department": "Finance"}
        )
    ]

    added = store.add_chunks(chunks)
    assert added == 2

    # Query compliance
    results = store.search("compliance background check", top_k=2)
    assert len(results) >= 1
    assert "compliance" in results[0].content.lower()

    # List documents
    docs = store.list_documents()
    assert len(docs) == 2

    # Delete document
    deleted = store.delete_document("compliance.pdf")
    assert deleted == 1

    remaining = store.list_documents()
    assert len(remaining) == 1
    assert remaining[0]["filename"] == "q3_report.docx"
