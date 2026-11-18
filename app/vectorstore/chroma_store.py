"""
ChromaDB Vector Store integration.
Provides persistent storage, semantic similarity search, and metadata filtering.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from app.config import config
from app.ingestion.base_loader import DocumentChunk
from app.vectorstore.embeddings import EmbeddingService
from app.utils.logger import logger


class VectorSearchResult:
    """Encapsulates a single retrieved chunk with its similarity score and metadata."""

    def __init__(self, chunk_id: str, content: str, score: float, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


class ChromaVectorStore:
    """Manages indexing, persistence, and semantic querying in ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.embedding_service = embedding_service or EmbeddingService()

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Enterprise AI Knowledge Base Vector Index"}
        )
        logger.info(f"Initialized ChromaDB at '{self.persist_directory}' (Collection: '{self.collection_name}')")

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Embeds and indexes document chunks in ChromaDB."""
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for idx, ch in enumerate(chunks):
            cid = ch.chunk_id or f"{ch.source_filename}_{ch.chunk_index}_{idx}"
            ids.append(cid)
            documents.append(ch.content)

            # Chroma metadata must have primitive values (str, int, float, bool)
            clean_meta = {}
            for k, v in ch.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_meta["source"] = ch.source_filename
            clean_meta["file_type"] = ch.file_type
            clean_meta["chunk_index"] = ch.chunk_index
            metadatas.append(clean_meta)

        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_service.embed_documents(documents)

        # Batch add to Chroma
        batch_size = 200
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        logger.info(f"Successfully indexed {len(ids)} chunks in ChromaDB.")
        return len(ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[VectorSearchResult]:
        """Executes a semantic similarity search against indexed document chunks."""
        count = self.collection.count()
        if count == 0:
            logger.warning("Search called on empty vector collection.")
            return []

        k = min(top_k, count)
        query_embedding = self.embedding_service.embed_query(query)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        raw_results = self.collection.query(**query_kwargs)

        results: List[VectorSearchResult] = []
        if raw_results and raw_results["ids"] and raw_results["ids"][0]:
            ids = raw_results["ids"][0]
            docs = raw_results["documents"][0]
            metas = raw_results["metadatas"][0]
            distances = raw_results["distances"][0] if "distances" in raw_results else [0.0] * len(ids)

            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                # Convert L2 distance or cosine distance to similarity score
                similarity = 1.0 / (1.0 + float(dist)) if dist is not None else 1.0
                if similarity >= min_score:
                    results.append(VectorSearchResult(
                        chunk_id=cid,
                        content=doc,
                        score=similarity,
                        metadata=meta or {}
                    ))

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns a list of unique ingested documents and their chunk counts."""
        count = self.collection.count()
        if count == 0:
            return []

        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        doc_stats: Dict[str, Dict[str, Any]] = {}

        for meta in all_meta:
            src = meta.get("source", "Unknown")
            ftype = meta.get("file_type", "unknown")
            if src not in doc_stats:
                doc_stats[src] = {
                    "filename": src,
                    "file_type": ftype,
                    "chunk_count": 0
                }
            doc_stats[src]["chunk_count"] += 1

        return list(doc_stats.values())

    def delete_document(self, filename: str) -> int:
        """Deletes all chunks associated with a specific document."""
        existing = self.collection.get(where={"source": filename})
        if existing and existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            logger.info(f"Deleted {len(existing['ids'])} chunks for file: {filename}")
            return len(existing["ids"])
        return 0

    def clear(self):
        """Clears all entries in the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Enterprise AI Knowledge Base Vector Index"}
        )
        logger.info("Cleared ChromaDB collection.")
