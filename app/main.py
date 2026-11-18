"""
Streamlit Web Application for Enterprise AI Knowledge Management.
Multi-document RAG with Multi-Agent Workflow (Planner, Retriever, Reasoning, Validator).
"""

import os
import shutil
from pathlib import Path
import streamlit as st

from app.config import config, UPLOAD_DIR
from app.ingestion.document_parser import DocumentParser
from app.vectorstore.chroma_store import ChromaVectorStore
from app.safety.sanitization import InputSanitizer
from app.safety.prompt_guard import PromptGuard
from app.agents.llm_client import LLMClient
from app.agents.orchestrator import AgentOrchestrator
from app.utils.logger import logger

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished enterprise look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
    }
    .agent-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .badge-verified {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-partial {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-unsupported {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
@st.cache_resource
def get_vector_store():
    return ChromaVectorStore()

@st.cache_resource
def get_parser():
    return DocumentParser()

vector_store = get_vector_store()
parser = get_parser()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = config.LLM_PROVIDER

if "api_key" not in st.session_state:
    st.session_state.api_key = config.GEMINI_API_KEY or config.OPENAI_API_KEY or ""


# -----------------------------------------------------------------------------
# Sidebar: Settings, Uploads & Document Library
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Hub Controls")

    # 1. LLM Provider Configuration
    with st.expander("🤖 LLM & Provider Settings", expanded=False):
        provider_options = ["gemini", "openai", "groq", "mock (offline engine)"]
        selected_provider = st.selectbox(
            "Select LLM Provider",
            options=provider_options,
            index=0 if config.LLM_PROVIDER in provider_options else 0
        )
        st.session_state.llm_provider = selected_provider

        user_api_key = st.text_input(
            "API Key (Optional / uses .env if blank)",
            type="password",
            value=st.session_state.api_key,
            help="Enter your API Key or leave empty to use the .env configuration or built-in offline engine."
        )
        if user_api_key:
            st.session_state.api_key = user_api_key

        top_k = st.slider("Top-K Retrieved Passages", min_value=1, max_value=10, value=config.TOP_K_RETRIEVAL)
        enable_validation = st.checkbox("Enable Hallucination Validator Agent", value=config.ENABLE_HALLUCINATION_VALIDATION)

    st.divider()

    # 2. Document Upload Section
    st.subheader("📁 Ingest Documents")
    st.caption("Supported: PDF, DOCX, TXT, CSV, Excel (.xlsx, .xls)")

    uploaded_files = st.file_uploader(
        "Upload enterprise knowledge files",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Process & Ingest Files", use_container_width=True):
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            clean_filename = InputSanitizer.sanitize_filename(uploaded_file.name)
            file_bytes = uploaded_file.getvalue()
            is_valid, val_err = InputSanitizer.validate_file(clean_filename, len(file_bytes))

            if not is_valid:
                st.error(f"Skipped '{uploaded_file.name}': {val_err}")
                continue

            # Save file to disk
            dest_path = UPLOAD_DIR / clean_filename
            with open(dest_path, "wb") as f:
                f.write(file_bytes)

            try:
                raw_doc, chunks = parser.parse_file(dest_path)
                # Remove existing chunks for file to avoid duplication
                vector_store.delete_document(clean_filename)
                vector_store.add_chunks(chunks)
                st.success(f"Ingested '{clean_filename}' ({len(chunks)} chunks)")
            except Exception as e:
                st.error(f"Error parsing '{clean_filename}': {e}")

            progress_bar.progress((i + 1) / total_files)

        st.rerun()

    st.divider()

    # 3. Document Library Overview
    st.subheader("📚 Ingested Library")
    docs = vector_store.list_documents()

    if docs:
        total_chunks = sum(d["chunk_count"] for d in docs)
        st.write(f"**Total Documents:** {len(docs)} | **Total Chunks:** {total_chunks}")

        for d in docs:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 **{d['filename']}** ({d['chunk_count']} chunks)")
            if col2.button("🗑️", key=f"del_{d['filename']}", help=f"Delete {d['filename']}"):
                vector_store.delete_document(d['filename'])
                st.rerun()

        if st.button("⚠️ Clear Entire Knowledge Base", type="secondary"):
            vector_store.clear()
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No documents indexed yet. Upload files above to begin.")


# -----------------------------------------------------------------------------
# Main Chat & Agent Interface
# -----------------------------------------------------------------------------
st.markdown(f'<div class="main-header">🧠 {config.APP_NAME}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Agent Enterprise RAG & Knowledge Synthesis '
    '(PDF, Word, TXT, CSV, Excel)</div>',
    unsafe_allow_html=True
)

# Metrics Strip
m1, m2, m3, m4 = st.columns(4)
total_docs = len(docs)
total_chunks = sum(d["chunk_count"] for d in docs)
m1.markdown(f'<div class="metric-card"><div class="metric-value">{total_docs}</div><div class="metric-label">Indexed Documents</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><div class="metric-value">{total_chunks}</div><div class="metric-label">Knowledge Chunks</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.llm_provider.split()[0].upper()}</div><div class="metric-label">Active Engine</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="metric-card"><div class="metric-value">ACTIVE</div><div class="metric-label">Safety Guardrails</div></div>', unsafe_allow_html=True)

st.write("")

# Initialize Orchestrator with current UI selections
llm_client = LLMClient(
    provider=st.session_state.llm_provider.split()[0],
    api_key=st.session_state.api_key
)
orchestrator = AgentOrchestrator(
    vector_store=vector_store,
    llm_client=llm_client
)

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display agent trace if available for assistant messages
        if msg["role"] == "assistant" and "trace" in msg:
            trace = msg.get("trace", {})
            citations = msg.get("citations", [])
            validation = msg.get("validation", {})
            plan = msg.get("plan", {})

            # 1. Citations
            if citations:
                with st.expander(f"📑 Sources & Citations ({len(citations)} references)", expanded=False):
                    for c in citations:
                        st.markdown(
                            f"**{c['label']}** — *Similarity: {c['score']:.2f}*\n"
                            f"> {c['content_preview']}"
                        )

            # 2. Agent Workflow Trace
            with st.expander("🤖 Agent Reasoning & Telemetry Trace", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**📋 Planner Agent**")
                    st.write(f"- Intent: `{plan.get('intent', 'General Q&A')}`")
                    st.write(f"- Subtasks: {plan.get('subtasks', ['Analyze', 'Retrieve', 'Synthesize'])}")
                with col_b:
                    st.markdown("**🛡️ Validator Agent**")
                    val_status = validation.get("status", "UNSUPPORTED")
                    badge_class = "badge-verified" if val_status == "VERIFIED" else ("badge-partial" if val_status == "PARTIALLY_VERIFIED" else "badge-unsupported")
                    st.markdown(f'Status: <span class="{badge_class}">{val_status}</span>', unsafe_allow_html=True)
                    st.write(f"Grounding Confidence: **{validation.get('confidence_score', 0.0) * 100:.1f}%**")
                    if validation.get("factuality_assessment"):
                        st.caption(f"Audit: {validation.get('factuality_assessment')}")

                st.markdown("**Execution Timers:**")
                telemetry = msg.get("workflow_trace", [])
                for t in telemetry:
                    st.caption(f"• **{t.get('agent', 'Agent')}** ({t.get('role', '')}): `{t.get('duration_ms', 0)} ms`")


# Quick Prompts if knowledge base is populated and no chat history yet
if not st.session_state.chat_history and total_docs > 0:
    st.markdown("**💡 Sample Questions you can ask:**")
    qc1, qc2, qc3 = st.columns(3)
    if qc1.button("📋 Summarize key points across all files"):
        st.session_state.pending_query = "Summarize the main themes and key findings from the uploaded documents."
        st.rerun()
    if qc2.button("📊 Analyze figures, numbers, and tabular data"):
        st.session_state.pending_query = "Extract and explain all key metrics, quantities, and financial or tabular values."
        st.rerun()
    if qc3.button("🔍 Find cross-document connections"):
        st.session_state.pending_query = "Compare the content across documents and highlight any commonalities or contrasts."
        st.rerun()

pending = st.session_state.pop("pending_query", None)
user_query = st.chat_input("Ask a question about your enterprise documents...") or pending

if user_query:
    # Append user question
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Execute Multi-Agent Workflow
    with st.chat_message("assistant"):
        with st.spinner("🤖 Multi-Agent System: Planning -> Retrieving -> Reasoning -> Validating..."):
            result = orchestrator.process_query(user_query, top_k=top_k)

        st.markdown(result["answer"])

        # Display Citations
        citations = result.get("citations", [])
        if citations:
            with st.expander(f"📑 Sources & Citations ({len(citations)} references)", expanded=True):
                for c in citations:
                    st.markdown(
                        f"**{c['label']}** — *Similarity: {c['score']:.2f}*\n"
                        f"> {c['content_preview']}"
                    )

        # Display Agent Workflow
        with st.expander("🤖 Agent Reasoning & Telemetry Trace", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**📋 Planner Agent**")
                st.write(f"- Intent: `{result['plan'].get('intent', 'General Q&A')}`")
                st.write(f"- Subtasks: {result['plan'].get('subtasks', [])}")
            with col_b:
                st.markdown("**🛡️ Validator Agent**")
                val_status = result["validation"].get("status", "UNSUPPORTED")
                badge_class = "badge-verified" if val_status == "VERIFIED" else ("badge-partial" if val_status == "PARTIALLY_VERIFIED" else "badge-unsupported")
                st.markdown(f'Status: <span class="{badge_class}">{val_status}</span>', unsafe_allow_html=True)
                st.write(f"Grounding Confidence: **{result['validation'].get('confidence_score', 0.0) * 100:.1f}%**")
                if result['validation'].get("factuality_assessment"):
                    st.caption(f"Audit: {result['validation'].get('factuality_assessment')}")

            st.markdown("**Agent Execution Timers:**")
            for t in result.get("workflow_trace", []):
                st.caption(f"• **{t.get('agent', 'Agent')}** ({t.get('role', '')}): `{t.get('duration_ms', 0)} ms`")

    # Save to session history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": citations,
        "plan": result.get("plan", {}),
        "validation": result.get("validation", {}),
        "workflow_trace": result.get("workflow_trace", [])
    })
