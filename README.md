# 🧠 Enterprise AI Knowledge Management Hub

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, Generative AI-powered Knowledge Management Application built with **Retrieval-Augmented Generation (RAG)**, **Vector Databases (ChromaDB)**, and an autonomous **Multi-Agent Orchestration Architecture** (Planner, Retriever, Reasoning, and Validator agents).

---

## 🚀 Key Capabilities

- 📄 **Multi-Format Document Ingestion**: Seamlessly ingest and index documents in **PDF**, **Word (.docx)**, **Plain Text (.txt, .md)**, **CSV**, and **Excel (.xlsx, .xls)**.
- 💬 **Natural Language Question Answering**: Query across massive enterprise documents with conversational context and high semantic recall.
- 🤖 **Autonomous Multi-Agent Workflow**:
  - **📋 Planner Agent**: Analyzes intent, constructs subtasks, generates targeted queries, and establishes document filters.
  - **🔍 Retriever Agent**: Queries ChromaDB vector store, deduplicates passages, ranks by cosine similarity, and formats exact citation references.
  - **🧠 Reasoning Agent**: Synthesizes cross-document context, performs tabular reasoning, and drafts cited responses.
  - **🛡️ Validator Agent**: Audits generated responses for hallucinations and evaluates grounding confidence against source documents.
- 📑 **Source Grounding & Citations**: Every answer provides clickable source cards with exact page numbers, sheet names, row indices, and relevance scores.
- 🔒 **Enterprise Safety & Guardrails**:
  - Built-in **Prompt Injection Guard** (detects jailbreaks and system override attempts).
  - Automated **PII Masking** (redacts emails, credit cards, SSNs, and API keys).
  - Path traversal and file size sanitization.
- 🔌 **Provider Agnostic**: Out-of-the-box support for **Google Gemini**, **OpenAI**, **Groq**, and a **Built-in Offline Reasoning Engine** that works without any external API keys!

---

## 🏗️ Architecture

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
                  |                     |  - SentenceTransformer dense index |
                  |                     +-----------------+------------------+
                  |                                       |
                  v                                       v
+----------------------------------------------------------------------------+
|                        Multi-Agent RAG Orchestrator                        |
|                                                                            |
|  [0] Prompt Guard & Safety  ──► Blocks jailbreaks & masks PII              |
|  [1] Planner Agent          ──► Query intent analysis & sub-task planning   |
|  [2] Retriever Agent        ──► Multi-query search & similarity ranking    |
|  [3] Reasoning Agent        ──► Context synthesis & cited answer drafting  |
|  [4] Validator Agent        ──► Factuality audit & confidence scoring      |
+----------------------------------------------------------------------------+
```

For complete architectural details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/AGENT_WORKFLOW.md](docs/AGENT_WORKFLOW.md).

---

## 📂 Project Structure

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
│   │   └── chroma_store.py      # Persistent ChromaDB storage & search engine
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

## ⚡ Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Clone and Setup Environment

```bash
git clone https://github.com/rayanakarthikeyan/AI-Knowledge-Management.git
cd AI-Knowledge-Management

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` to add your preferred LLM provider and API key:
```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Note: You can also leave this blank and use the built-in Offline Reasoning Engine or enter your API key dynamically in the Streamlit UI).*

### 4. Launch the Application

```bash
streamlit run app/main.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the comprehensive pytest suite:
```bash
pytest -v
```

All 12 unit and integration tests validate the document loaders, chunking logic, vector database, prompt injection guard, and multi-agent orchestrator.

---

## 🐳 Docker Deployment

To build and run with Docker:

```bash
docker build -t enterprise-ai-knowledge-hub .
docker run -p 8501:8501 --env-file .env enterprise-ai-knowledge-hub
```

Or with Docker Compose:
```bash
docker compose up -d
```

---

## 🛡️ Safety & Compliance

- **Prompt Injection Defense**: Evaluates incoming queries against adversarial jailbreak signatures.
- **PII Redaction**: Detects and redacts Credit Cards, SSNs, and Email addresses before displaying or storing data.
- **Data Grounding**: The Validator Agent audits every response to guarantee claims originate from indexed documents.

---

## 📄 License
MIT License. Built for enterprise knowledge intelligence.
