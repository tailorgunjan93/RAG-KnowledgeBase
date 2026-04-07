"""
IVectorStore — Abstract contract for any vector database.

Plug-and-Play: To swap FAISS for Chroma or Pinecone, create a new class
that implements this interface. Only container.py needs to change.
"""
from abc import ABC, abstractmethod
from langchain_core.documents import Document


class IVectorStore(ABC):
    """
    Abstract base for all vector store providers.
    Implementations: FAISSVectorStore, ChromaVectorStore, PineconeVectorStore, etc.
    """

    @abstractmethod
    def add_documents(self, docs: list[Document]) -> None:
        """
        Add or upsert documents into the vector store.

        Args:
            docs: Chunked LangChain Document objects to embed and store.
        """
        ...

    @abstractmethod
    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """
        Perform a semantic similarity search.

        Args:
            query: Natural language query string.
            k: Maximum number of results to return.

        Returns:
            List of (Document, score) tuples. Score is a float (lower = more similar for L2).
        """
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist the vector store to disk (no-op for cloud stores)."""
        ...
