"""
FAISSVectorStore — Concrete implementation of IVectorStore using Facebook FAISS.

Wraps all FAISS index operations from the legacy embeddings.py and faiss_search.py
into a single, cohesive, testable class.

To swap for Chroma: create ChromaVectorStore(IVectorStore) and update container.py.
"""
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from Src.Config.settings import Settings
from Src.Interfaces.IVectorStore import IVectorStore


class FAISSVectorStore(IVectorStore):
    """
    FAISS-backed vector store with per-document-set index management.

    Each ingested file gets its own named FAISS index directory under
    the configured vector_store_base_path. This allows per-file granularity
    for the dynamic index routing logic.

    Usage:
        store = FAISSVectorStore(settings, index_name="my_document")
        store.add_documents(docs)
        results = store.search("what is attention?")
    """

    def __init__(self, settings: Settings, index_name: str) -> None:
        """
        Args:
            settings: Application settings (for paths and embedding model name).
            index_name: Subdirectory name for this index (usually the file stem).
        """
        self._settings = settings
        self._index_name = index_name
        self._index_path = Path(settings.vector_store_base_path) / index_name
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self._index: FAISS | None = None

    # ── IVectorStore implementation ──────────────────────────────────────────

    def add_documents(self, docs: list[Document]) -> None:
        """Embed and add documents. Creates a new index if none exists."""
        if self._index_path.exists():
            self._index = self._load()
            self._index.add_documents(docs)
        else:
            self._index = FAISS.from_documents(docs, embedding=self._embeddings)
        self.save()

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """
        Semantic similarity search. Returns Python floats (not numpy.float32)
        to ensure LangGraph MemorySaver serialization compatibility.
        """
        index = self._ensure_loaded()
        results = index.similarity_search_with_score(query, k=k)
        # Convert numpy.float32 → float to prevent msgpack serialization errors
        return [(doc, float(score)) for doc, score in results]

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        if self._index is not None:
            self._index_path.mkdir(parents=True, exist_ok=True)
            self._index.save_local(str(self._index_path))

    # ── Public helper ────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        """Return the total number of vectors stored in this index."""
        index = self._ensure_loaded()
        return index.index.ntotal

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load(self) -> FAISS:
        return FAISS.load_local(
            folder_path=str(self._index_path),
            embeddings=self._embeddings,
            index_name="index",
            allow_dangerous_deserialization=True,
        )

    def _ensure_loaded(self) -> FAISS:
        if self._index is None:
            self._index = self._load()
        return self._index
