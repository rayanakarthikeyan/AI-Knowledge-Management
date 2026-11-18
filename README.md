# Enterprise AI Knowledge Management Hub

An enterprise-grade, Generative AI-powered Knowledge Management Application built with **Retrieval-Augmented Generation (RAG)**, **Vector Databases (ChromaDB)**, and an autonomous **Multi-Agent Orchestration Architecture** (Planner, Retriever, Reasoning, and Validator agents).

---

## Key Capabilities

- **Multi-Format Document Ingestion**: Ingest and index documents in PDF, Word (.docx), Plain Text (.txt, .md), CSV, and Excel (.xlsx, .xls).
- **Natural Language Question Answering**: Query across enterprise document repositories with conversational context and hybrid retrieval precision.
- **Autonomous Multi-Agent Workflow**:
  - **Planner Agent**: Analyzes intent, constructs subtasks, generates targeted queries, and establishes document filters.
  - **Retriever Agent**: Executes hybrid retrieval (dense vector similarity + lexical keyword matching with Reciprocal Rank Fusion) over ChromaDB, deduplicates passages, and formats exact citation references.
  - **Reasoning Agent**: Synthesizes cross-document context, performs tabular reasoning, and drafts cited responses.
  - **Validator Agent**: Audits generated responses for hallucinations and evaluates grounding confidence against source documents.
- **Source Grounding & Citations**: Every answer provides verified source references with exact page numbers, sheet names, row indices, and relevance scores.
- **Enterprise Safety & Guardrails**:
  - Built-in **Prompt Injection Guard** (detects jailbreaks and system override attempts).
  - Automated **PII Masking** (redacts emails, credit cards, SSNs, and API keys).
  - Path traversal and file size sanitization.
- **Provider Agnostic**: Support for Google Gemini, OpenAI, Groq, and a built-in Offline Reasoning Engine.

---

## Architecture

```
                                  +------------------------------------+
                                  |    Enterprise Document Sources     |
                                  |  (PDF, Word, TXT, CSV, Excel)      |
                                  +-----------------+------------------+
                                                    |
                                                    v
+-----------------------------------+   +------------------------------------+
|       Streamlit Web Interface     |   |   Document Parser & Sanitization   |
|  - Ingestion Hub                  |   |    - File type & size validation   |
|  - Real-time Agent Reasoning Trace|   |    - Page & table cell extraction  |
|  - Grounded Citations & Confidence|   +-----------------+------------------+
+-----------------+-----------------+                     |
                  |                                       v
                  |                     +------------------------------------+
                  |                     |        Document Chunker            |
                  |                     |  - Sliding window with overlap     |
                  |                     |  - Page & row metadata inheritance |
                  |                     +-----------------+------------------+
                  |                                       |
                  |                                       v
                  |                     +------------------------------------+
                  |                     |  ChromaDB Vector Store & Embedding |
                  |                     |  - Hybrid Semantic + Lexical RRF   |
                  |                     +-----------------+------------------+
                  |                                       |
                  v                                       v
+----------------------------------------------------------------------------+
|                        Multi-Agent RAG Orchestrator                        |
|                                                                            |
|  [0] Prompt Guard & Safety  --> Blocks jailbreaks & masks PII              |
|  [1] Planner Agent          --> Query intent analysis & sub-task planning   |
|  [2] Retriever Agent        --> Hybrid multi-query search & ranking        |
|  [3] Reasoning Agent        --> Context synthesis & cited answer drafting  |
|  [4] Validator Agent        --> Factuality audit & confidence scoring      |
+----------------------------------------------------------------------------+
```

For complete architectural details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md).

---

## Project Structure

```
AI Knowledge Management/
│
├── app/
│   ├── main.py                  # Streamlit application UI & chat interface
│   ├── config.py                # Central app configuration & thresholds
│   │
│   ├── ingestion/               # Multi-format document ingestion pipeline
│   │   ├── base_loader.py       # BaseLoader and DocumentChunk definitions
│   │   ├── pdf_loader.py        # PDF text & page extractor (pypdf)
│   │   ├── docx_loader.py       # DOCX paragraphs & table parser (python-docx)
│   │   ├── txt_loader.py        # Plain text & markdown multi-encoding parser
│   │   ├── tabular_loader.py    # CSV & Excel row-wise structured extractor
│   │   ├── chunker.py           # Overlapping chunker with metadata retention
│   │   └── document_parser.py   # Universal parser router
│   │
│   ├── vectorstore/             # Embeddings & Vector Indexing
│   │   ├── embeddings.py        # Dense embeddings (SentenceTransformers / fallback)
│   │   └── chroma_store.py      # Hybrid retrieval ChromaDB storage & search
│   │
│   ├── agents/                  # Autonomous Multi-Agent Pipeline
│   │   ├── base_agent.py        # Base agent execution with telemetry timers
│   │   ├── llm_client.py        # Unified Gemini / OpenAI / Groq / Offline client
│   │   ├── planner_agent.py     # Decomposes queries into execution subtasks
│   │   ├── retriever_agent.py   # Fetches and ranks context passages
│   │   ├── reasoning_agent.py   # Synthesizes cited context-aware answers
│   │   ├── validator_agent.py   # Audits hallucinations & calculates confidence
│   │   └── orchestrator.py      # Multi-agent coordination pipeline
│   │
│   ├── safety/                  # Enterprise Security & Guardrails
│   │   ├── prompt_guard.py      # Prompt injection detection & PII redactor
│   │   └── sanitization.py      # Filename & file size sanitization
│   │
│   └── utils/
│       └── logger.py            # Structured logging utility
│
├── docs/
│   ├── ARCHITECTURE.md          # In-depth architectural design
│   └── AGENT_WORKFLOW.md        # Detailed agent protocols & JSON schemas
│
├── sample_documents/            # Enterprise test files for all supported formats
│   ├── enterprise_security_compliance.pdf
│   ├── enterprise_ai_policy.txt
│   ├── executive_summary.docx
│   ├── quarterly_financials.csv
│   └── project_budgets.xlsx
│
├── tests/                       # Complete automated unit test suite
│   ├── test_loaders.py          # PDF, DOCX, TXT, CSV, Excel loader tests
│   ├── test_chunker.py          # Chunk boundary and metadata tests
│   ├── test_vectorstore.py      # ChromaDB storage & search tests
│   ├── test_safety.py           # Prompt injection and PII masking tests
│   └── test_agents.py           # Multi-agent workflow and validation tests
│
├── .env.example                 # Environment variables template
├── .gitignore                   # Ignore rules for models, vector data, venvs
├── requirements.txt             # Locked Python dependencies
├── Dockerfile                   # Containerization configuration
├── docker-compose.yml           # Multi-container orchestration
└── README.md                    # Project documentation
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Setup Environment

```bash
git clone https://github.com/rayanakarthikeyan/AI-Knowledge-Management.git
cd AI-Knowledge-Management

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Configure your preferred LLM provider in `.env`:
```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
```
*(You can also use the built-in Offline Reasoning Engine without any external API keys).*

### 4. Launch Application

```bash
streamlit run app/main.py
```

Access the interface at `http://localhost:8501`.

---

## Automated Verification Tests

Run the complete test suite:
```bash
pytest -v
```

All unit and integration tests validate the document loaders, chunking logic, vector database, prompt injection guard, and multi-agent orchestrator.

---

## Docker Deployment

Build and run with Docker:

```bash
docker build -t enterprise-ai-knowledge-hub .
docker run -p 8501:8501 --env-file .env enterprise-ai-knowledge-hub
```

Or with Docker Compose:
```bash
docker compose up -d
```

---

## License
MIT License. Built for enterprise knowledge intelligence.
