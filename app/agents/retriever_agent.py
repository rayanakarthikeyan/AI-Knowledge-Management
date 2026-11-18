"""
Retriever Agent.
Executes semantic vector searches, applies metadata filtering, and ranks relevant passages.
"""

from typing import Any, Dict, List
from app.agents.base_agent import BaseAgent
from app.vectorstore.chroma_store import ChromaVectorStore, VectorSearchResult
from app.config import config
from app.utils.logger import logger


class RetrieverAgent(BaseAgent):
    """Executes multi-query retrieval and deduplication over the knowledge base."""

    def __init__(self, vector_store: ChromaVectorStore):
        super().__init__(
            name="RetrieverAgent",
            role="Semantic Document Retrieval & Passage Ranking"
        )
        self.vector_store = vector_store

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        plan = input_data.get("plan", {})
        search_queries = plan.get("search_queries", [])
        original_query = input_data.get("query", "")
        top_k = input_data.get("top_k", config.TOP_K_RETRIEVAL)
        doc_filter = plan.get("document_filter")

        if not search_queries:
            search_queries = [original_query]
        elif original_query not in search_queries:
            search_queries.append(original_query)

        filter_metadata = None
        if doc_filter and isinstance(doc_filter, str) and doc_filter.strip().lower() != "null":
            filter_metadata = {"source": doc_filter.strip()}

        seen_chunk_ids = set()
        all_results: List[VectorSearchResult] = []

        for q in search_queries:
            results = self.vector_store.search(
                query=q,
                top_k=top_k,
                filter_metadata=filter_metadata,
                min_score=config.SIMILARITY_THRESHOLD
            )
            for res in results:
                if res.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(res.chunk_id)
                    all_results.append(res)

        # Sort combined results descending by similarity score
        all_results.sort(key=lambda x: x.score, reverse=True)
        top_results = all_results[:top_k]

        # Format context string for reasoning agent
        context_blocks = []
        citations = []

        for idx, item in enumerate(top_results, start=1):
            src = item.metadata.get("source", "Document")
            page = item.metadata.get("page_number")
            sheet = item.metadata.get("sheet_name")

            loc = f"Page {page}" if page else (f"Sheet '{sheet}'" if sheet else "General")
            citation_label = f"[{idx}] {src} ({loc})"

            context_blocks.append(f"--- SOURCE {citation_label} ---\n{item.content}")
            citations.append({
                "index": idx,
                "label": citation_label,
                "source": src,
                "location": loc,
                "score": item.score,
                "content_preview": item.content[:200] + ("..." if len(item.content) > 200 else ""),
                "chunk_id": item.chunk_id
            })

        formatted_context = "\n\n".join(context_blocks) if context_blocks else "NO_RELEVANT_CONTEXT_FOUND"

        return {
            "retrieved_count": len(top_results),
            "citations": citations,
            "formatted_context": formatted_context,
            "raw_results": [r.to_dict() for r in top_results]
        }
