"""
Enterprise Knowledge Management Platform
Streamlit Web Interface with Autonomous Multi-Agent RAG Orchestration.
"""

import json
from pathlib import Path
import streamlit as st

from app.config import config, BASE_DIR, UPLOAD_DIR
from app.ingestion.document_parser import DocumentParser
from app.vectorstore.chroma_store import ChromaVectorStore
from app.safety.sanitization import InputSanitizer
from app.safety.prompt_guard import PromptGuard
from app.agents.llm_client import LLMClient
from app.agents.orchestrator import AgentOrchestrator
from app.utils.logger import logger

SAMPLE_DIR = BASE_DIR / "sample_documents"

# -----------------------------------------------------------------------------
# Page Configuration & Professional Corporate Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Typography & Palette */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1E293B;
    }
    .header-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-bottom: 0.15rem;
    }
    .header-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }
    /* Executive Metric Card */
    .metric-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .metric-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-title {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
        font-weight: 600;
    }
    /* Status Badges */
    .status-badge-verified {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .status-badge-partial {
        display: inline-block;
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .status-badge-unsupported {
        display: inline-block;
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
    }
    .citation-header {
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .citation-snippet {
        color: #475569;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Resource Caching
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
# Helper: Export Conversation Report
# -----------------------------------------------------------------------------
def generate_audit_export() -> str:
    """Formats the current session history into a Markdown compliance audit report."""
    lines = [
        "# Enterprise Knowledge Management - Audit & Telemetry Report",
        f"Generated Timestamp: {st.session_state.get('last_query_time', 'N/A')}",
        f"Active Model Engine: {st.session_state.llm_provider}",
        "---",
        ""
    ]
    for idx, item in enumerate(st.session_state.chat_history, start=1):
        if item["role"] == "user":
            lines.append(f"### Inquiry {idx}: {item['content']}\n")
        else:
            lines.append(f"**Synthesized Answer:**\n{item['content']}\n")
            validation = item.get("validation", {})
            lines.append(f"- **Verification Status:** {validation.get('status', 'N/A')}")
            lines.append(f"- **Grounding Confidence:** {validation.get('confidence_score', 0.0) * 100:.1f}%")
            lines.append(f"- **Factuality Assessment:** {validation.get('factuality_assessment', 'N/A')}")

            citations = item.get("citations", [])
            if citations:
                lines.append("\n**Citations:**")
                for c in citations:
                    lines.append(f"- {c['label']} (Similarity: {c['score']:.3f})")
            lines.append("\n---\n")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Sidebar: Control Panel & Document Management
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### System Configuration")

    # 1. Model & Engine Controls
    with st.expander("Inference Engine & Parameters", expanded=False):
        provider_options = ["gemini", "openai", "groq", "offline reasoning engine"]
        current_idx = 0
        for idx, opt in enumerate(provider_options):
            if opt.startswith(config.LLM_PROVIDER):
                current_idx = idx
                break

        selected_provider = st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=current_idx
        )
        st.session_state.llm_provider = selected_provider

        user_api_key = st.text_input(
            "API Key (Overrides .env)",
            type="password",
            value=st.session_state.api_key,
            help="Leave blank to use environment variable or built-in offline engine."
        )
        if user_api_key:
            st.session_state.api_key = user_api_key

        top_k = st.slider("Retrieval Candidate Limit (Top-K)", min_value=1, max_value=10, value=config.TOP_K_RETRIEVAL)
        enable_validation = st.checkbox("Autonomous Hallucination Audit", value=config.ENABLE_HALLUCINATION_VALIDATION)

    st.divider()

    # 2. Document Ingestion
    st.markdown("### Document Ingestion")
    st.caption("Accepted Formats: PDF, DOCX, TXT, CSV, Excel (.xlsx, .xls)")

    uploaded_files = st.file_uploader(
        "Select documents to index",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and st.button("Process & Index Documents", use_container_width=True):
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            clean_filename = InputSanitizer.sanitize_filename(uploaded_file.name)
            file_bytes = uploaded_file.getvalue()
            is_valid, val_err = InputSanitizer.validate_file(clean_filename, len(file_bytes))

            if not is_valid:
                st.error(f"Rejected '{uploaded_file.name}': {val_err}")
                continue

            dest_path = UPLOAD_DIR / clean_filename
            with open(dest_path, "wb") as f:
                f.write(file_bytes)

            try:
                raw_doc, chunks = parser.parse_file(dest_path)
                vector_store.delete_document(clean_filename)
                vector_store.add_chunks(chunks)
                st.success(f"Indexed: {clean_filename} ({len(chunks)} chunks)")
            except Exception as e:
                st.error(f"Processing error for '{clean_filename}': {e}")

            progress_bar.progress((i + 1) / total_files)

        st.rerun()

    # Convenience helper: Load Sample Documents
    if SAMPLE_DIR.exists():
        if st.button("Load Bundled Sample Documents", help="Loads pre-configured enterprise PDF, DOCX, TXT, CSV, and Excel sample files", use_container_width=True):
            sample_files = list(SAMPLE_DIR.glob("*.*"))
            loaded_count = 0
            for sf in sample_files:
                if sf.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                    try:
                        raw_doc, chunks = parser.parse_file(sf)
                        vector_store.delete_document(sf.name)
                        vector_store.add_chunks(chunks)
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"Error loading sample file {sf.name}: {e}")
            if loaded_count > 0:
                st.success(f"Successfully loaded {loaded_count} sample enterprise documents.")
                st.rerun()

    st.divider()

    # 3. Knowledge Repository Overview
    st.markdown("### Document Repository")
    docs = vector_store.list_documents()

    if docs:
        total_chunks = sum(d["chunk_count"] for d in docs)
        st.write(f"Total Files: **{len(docs)}** | Total Chunks: **{total_chunks}**")

        for d in docs:
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{d['filename']}**  \n`{d['file_type'].upper()}` • {d['chunk_count']} chunks")
            if c2.button("Remove", key=f"del_{d['filename']}"):
                vector_store.delete_document(d['filename'])
                st.rerun()

        st.write("")
        col_clear1, col_clear2 = st.columns(2)
        if col_clear1.button("Reset KB", type="secondary", use_container_width=True):
            vector_store.clear()
            st.session_state.chat_history = []
            st.rerun()

        if col_clear2.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No documents indexed. Upload files above or load bundled samples.")


# -----------------------------------------------------------------------------
# Main Application Dashboard
# -----------------------------------------------------------------------------
st.markdown('<div class="header-title">Enterprise Knowledge Management Hub</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-subtitle">Multi-Agent Retrieval-Augmented Generation & Cross-Document Synthesis Engine</div>',
    unsafe_allow_html=True
)

# Executive Metric Strip
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
total_docs = len(docs)
total_chunks = sum(d["chunk_count"] for d in docs)
active_engine = st.session_state.llm_provider.split()[0].upper()

col_m1.markdown(f'<div class="metric-panel"><div class="metric-number">{total_docs}</div><div class="metric-title">Indexed Documents</div></div>', unsafe_allow_html=True)
col_m2.markdown(f'<div class="metric-panel"><div class="metric-number">{total_chunks}</div><div class="metric-title">Knowledge Chunks</div></div>', unsafe_allow_html=True)
col_m3.markdown(f'<div class="metric-panel"><div class="metric-number">{active_engine}</div><div class="metric-title">Active LLM Engine</div></div>', unsafe_allow_html=True)
col_m4.markdown(f'<div class="metric-panel"><div class="metric-number">ENFORCED</div><div class="metric-title">Security Guardrails</div></div>', unsafe_allow_html=True)

st.write("")

# Initialize Orchestrator
llm_client = LLMClient(
    provider=st.session_state.llm_provider.split()[0],
    api_key=st.session_state.api_key
)
orchestrator = AgentOrchestrator(
    vector_store=vector_store,
    llm_client=llm_client
)

# Render Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "citations" in msg:
            citations = msg.get("citations", [])
            validation = msg.get("validation", {})
            plan = msg.get("plan", {})
            trace = msg.get("workflow_trace", [])

            # Source Citations
            if citations:
                with st.expander(f"Grounded Citations ({len(citations)} references)", expanded=False):
                    for c in citations:
                        st.markdown(
                            f'<div class="citation-card">'
                            f'<div class="citation-header">{c["label"]} — Similarity Score: {c["score"]:.3f} [{c.get("match_type", "hybrid").upper()}]</div>'
                            f'<div class="citation-snippet">"{c["content_preview"]}"</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Agent Workflow & Audit Telemetry
            with st.expander("Agent Reasoning & Audit Trace", expanded=False):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Planner Agent Decomposition**")
                    st.write(f"- Strategic Intent: `{plan.get('intent', 'General Synthesis')}`")
                    st.write(f"- Planned Subtasks: {plan.get('subtasks', [])}")
                    if plan.get("document_filter"):
                        st.write(f"- Document Scope Filter: `{plan.get('document_filter')}`")

                with cb:
                    st.markdown("**Validator Agent Audit**")
                    vstatus = validation.get("status", "UNSUPPORTED")
                    badge_style = "status-badge-verified" if vstatus == "VERIFIED" else ("status-badge-partial" if vstatus == "PARTIALLY_VERIFIED" else "status-badge-unsupported")
                    st.markdown(f'Verification Status: <span class="{badge_style}">[{vstatus}]</span>', unsafe_allow_html=True)
                    st.write(f"Grounding Confidence Score: **{validation.get('confidence_score', 0.0) * 100:.1f}%**")
                    if validation.get("factuality_assessment"):
                        st.caption(f"Audit Summary: {validation.get('factuality_assessment')}")

                st.markdown("**Execution Latency Telemetry:**")
                for t in trace:
                    st.caption(f"- **{t.get('agent', 'Agent')}** ({t.get('role', '')}): `{t.get('duration_ms', 0)} ms`")


# Quick Sample Queries
if not st.session_state.chat_history and total_docs > 0:
    st.markdown("**Suggested Queries:**")
    qc1, qc2, qc3 = st.columns(3)
    if qc1.button("Summarize governance & security policies"):
        st.session_state.pending_query = "Summarize the key information security and AI governance policies from the uploaded documents."
        st.rerun()
    if qc2.button("Analyze financial & tabular allocations"):
        st.session_state.pending_query = "Extract all quarterly revenue, net profit, and project budget allocations across the uploaded spreadsheets."
        st.rerun()
    if qc3.button("Conduct cross-document audit"):
        st.session_state.pending_query = "Compare initiatives outlined in the executive summary against the technical specifications."
        st.rerun()

pending = st.session_state.pop("pending_query", None)
user_query = st.chat_input("Enter natural language inquiry...") or pending

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Executing Multi-Agent Workflow: Planning -> Retrieval -> Reasoning -> Validation..."):
            result = orchestrator.process_query(user_query, top_k=top_k)

        st.markdown(result["answer"])

        citations = result.get("citations", [])
        if citations:
            with st.expander(f"Grounded Citations ({len(citations)} references)", expanded=True):
                for c in citations:
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<div class="citation-header">{c["label"]} — Similarity Score: {c["score"]:.3f} [{c.get("match_type", "hybrid").upper()}]</div>'
                        f'<div class="citation-snippet">"{c["content_preview"]}"</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with st.expander("Agent Reasoning & Audit Trace", expanded=True):
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Planner Agent Decomposition**")
                st.write(f"- Strategic Intent: `{result['plan'].get('intent', 'General Synthesis')}`")
                st.write(f"- Planned Subtasks: {result['plan'].get('subtasks', [])}")
                if result['plan'].get("document_filter"):
                    st.write(f"- Document Scope Filter: `{result['plan'].get('document_filter')}`")

            with cb:
                st.markdown("**Validator Agent Audit**")
                vstatus = result["validation"].get("status", "UNSUPPORTED")
                badge_style = "status-badge-verified" if vstatus == "VERIFIED" else ("status-badge-partial" if vstatus == "PARTIALLY_VERIFIED" else "status-badge-unsupported")
                st.markdown(f'Verification Status: <span class="{badge_style}">[{vstatus}]</span>', unsafe_allow_html=True)
                st.write(f"Grounding Confidence Score: **{result['validation'].get('confidence_score', 0.0) * 100:.1f}%**")
                if result['validation'].get("factuality_assessment"):
                    st.caption(f"Audit Summary: {result['validation'].get('factuality_assessment')}")

            st.markdown("**Execution Latency Telemetry:**")
            for t in result.get("workflow_trace", []):
                st.caption(f"- **{t.get('agent', 'Agent')}** ({t.get('role', '')}): `{t.get('duration_ms', 0)} ms`")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": citations,
        "plan": result.get("plan", {}),
        "validation": result.get("validation", {}),
        "workflow_trace": result.get("workflow_trace", [])
    })

# Export Compliance & Audit Report Option
if st.session_state.chat_history:
    st.divider()
    audit_report_text = generate_audit_export()
    st.download_button(
        label="Download Compliance & Audit Report",
        data=audit_report_text,
        file_name="enterprise_knowledge_audit_report.md",
        mime="text/markdown",
        help="Export all queries, synthesized answers, citations, and agent audit metrics."
    )
