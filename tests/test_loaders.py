"""
Unit tests for Document Loaders (PDF, DOCX, TXT, CSV, Excel).
"""

from pathlib import Path
import pandas as pd
import pytest
from app.ingestion.txt_loader import TxtLoader
from app.ingestion.tabular_loader import TabularLoader
from app.ingestion.document_parser import DocumentParser


def test_txt_loader(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Paragraph one of enterprise policy.\n\nParagraph two with financial metrics.", encoding="utf-8")

    loader = TxtLoader()
    doc = loader.load(txt_file)

    assert doc.source_filename == "sample.txt"
    assert doc.file_type == "txt"
    assert "Paragraph one" in doc.content
    assert len(doc.pages_or_sections) == 2


def test_csv_loader(tmp_path: Path):
    csv_file = tmp_path / "sales.csv"
    df = pd.DataFrame({
        "Department": ["Sales", "Engineering", "Marketing"],
        "Budget": [50000, 120000, 35000],
        "Headcount": [5, 12, 3]
    })
    df.to_csv(csv_file, index=False)

    loader = TabularLoader()
    doc = loader.load(csv_file)

    assert doc.source_filename == "sales.csv"
    assert doc.file_type == "tabular"
    assert "Department: Engineering" in doc.content
    assert doc.metadata["total_rows"] == 3


def test_excel_loader(tmp_path: Path):
    excel_file = tmp_path / "inventory.xlsx"
    df = pd.DataFrame({
        "SKU": ["A101", "B202"],
        "Stock": [150, 42],
        "Warehouse": ["NY", "CA"]
    })
    df.to_excel(excel_file, index=False)

    loader = TabularLoader()
    doc = loader.load(excel_file)

    assert doc.source_filename == "inventory.xlsx"
    assert doc.file_type == "tabular"
    assert "SKU: A101" in doc.content
    assert "Stock: 150" in doc.content


def test_document_parser_routing(tmp_path: Path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Antigravity Knowledge Note", encoding="utf-8")

    parser = DocumentParser()
    raw_doc, chunks = parser.parse_file(txt_file)

    assert raw_doc.source_filename == "notes.txt"
    assert len(chunks) >= 1
    assert "Antigravity Knowledge Note" in chunks[0].content
