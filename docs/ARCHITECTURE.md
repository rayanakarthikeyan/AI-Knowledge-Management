# Enterprise Generative AI Knowledge Management Architecture

## 1. Executive Summary

The **Enterprise Generative AI Knowledge Management Application** is a high-performance, modular RAG (Retrieval-Augmented Generation) system built to ingest, index, synthesize, and reason over heterogeneous enterprise document repositories (PDF, DOCX, TXT, CSV, and Excel).

The solution utilizes a decoupled multi-agent architecture where individual agents are responsible for distinct cognitive tasks: **Strategic Planning**, **Semantic & Tabular Retrieval**, **Contextual Reasoning & Synthesis**, and **Factuality / Hallucination Verification**.

---

## 2. High-Level Architecture Diagram

```
                                  +------------------------------------+
                                  |     Enterprise Document Sources    |
                                  |   (PDF, Word, TXT, CSV, Excel)     |
                                  +-----------------+------------------+
                                                    |
                                                    v
+-----------------------------------+   +------------------------------------+
|       Streamlit Web Interface     |   |   Document Parser & Sanitization   |
|  - Multi-File Drag & Drop         |   |    - File boundary & MIME validation
|  - Real-time Agent Reasoning Trace|   |    - Multi-encoding TXT parser     |
|  - Verified Answer & Citations    |   |    - Tabular cell & row extractor  |
|  - Knowledge Base Manager         |   |    - PDF & DOCX section extractors |
+-----------------+-----------------+   +-----------------+------------------+
                  |                                       |
                  |                                       v
                  |                     +------------------------------------+
                  |                     |        Document Chunker            |
                  |                     |  - Sliding window with overlap     |
                  |                     |  - Page & row metadata preservation|
                  |                     +-----------------+------------------+
                  |                                       |
                  |                                       v
                  |                     +------------------------------------+
                  |                     |  ChromaDB Vector Store & Embedding |
                  |                     |  - Dense 384-d semantic vectors    |
                  |                     |  - Metadata filtering by document  |
                  |                     +-----------------+------------------+
                  |                                       |
                  v                                       v
+----------------------------------------------------------------------------+
|                        Multi-Agent RAG Orchestrator                        |
|                                                                            |
|  [0] Prompt Guard & Safety:                                                |
|      - Jailbreak & prompt injection detection                              |
|      - PII masking (emails, credit cards, SSN, API keys)                   |
|                                                                            |
|  [1] Planner Agent:                                                        |
|      - Decomposes query into intent, subtasks, target search queries       |
|                                                                            |
|  [2] Retriever Agent:                                                      |
|      - Multi-query similarity search against ChromaDB                      |
|      - Deduplication, score ranking, and citation tagging                  |
|                                                                            |
|  [3] Reasoning Agent:                                                      |
|      - Cross-document synthesis & tabular reasoning                        |
|      - Direct grounding and in-line citations [1], [2]                     |
|                                                                            |
|  [4] Validator Agent:                                                      |
|      - Audits generated claims against retrieved context                   |
|      - Outputs status: VERIFIED, PARTIALLY_VERIFIED, UNSUPPORTED           |
|      - Computes quantitative grounding confidence score                    |
+----------------------------------------------------------------------------+
```

---

## 3. Ingestion & Preprocessing Subsystems

### 3.1 Supported Document Formats
| Format | Loader Module | Strategy |
| :--- | :--- | :--- |
| **PDF** | `PDFLoader` (`pypdf`) | Extracts text per page, captures exact page numbers for citation attribution. |
| **Word (DOCX)** | `DocxLoader` (`python-docx`) | Extracts paragraphs, headings (`### Heading`), and structured table matrices. |
| **Plain Text (TXT, MD)** | `TxtLoader` | Detects encodings (UTF-8, Latin-1, CP1252), parses paragraph sections. |
| **CSV & Excel (XLSX, XLS)** | `TabularLoader` (`pandas`, `openpyxl`) | Converts sheets and rows into structured semantic strings (`[Sheet - Row X] Col: Val`), preserving tabular relationships for embedding lookup. |

### 3.2 Semantic Chunking
- **Sliding Window**: Chunks are generated with configurable character limits (`CHUNK_SIZE`, default 600 chars) and overlap (`CHUNK_OVERLAP`, default 100 chars).
- **Boundary Snapping**: Snaps to sentence ends, newlines, or spaces to prevent mid-word truncations.
- **Metadata Inheritance**: Every chunk inherits source filename, file type, page number or sheet/row index, and a unique chunk ID.

---

## 4. Vector Storage & Semantic Search

- **Database**: ChromaDB persistent client (`./data/vector_store`).
- **Embedding Pipeline**: `SentenceTransformer` (`all-MiniLM-L6-v2`) with a deterministic normalized fallback engine to ensure 100% offline uptime.
- **Metadata Filtering**: Enables scoping queries to specific documents (e.g., `"in the sales spreadsheet"`) via Chroma's `where` clause.
- **Score Conversion**: Converts L2 distance to similarity score \( \text{Score} = \frac{1}{1 + \text{distance}} \).

---

## 5. Security Guardrails & Error Boundaries

1. **Prompt Injection Prevention**:
   - Compiles regex patterns blocking system prompt extraction, jailbreaks, and instructions override.
2. **PII Masking**:
   - Scans text for Social Security Numbers (SSN), Credit Cards, and Email addresses, masking them as `[REDACTED_*]`.
3. **File Validation**:
   - Rejects unallowed extensions and restricts file size up to 50MB.
   - Cleans filenames to prevent path traversal (`../`).
