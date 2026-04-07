"""
SearchService — Orchestrates retrieval from vector and graph stores.

SRP: This service ONLY coordinates multi-source document retrieval.
It knows nothing about FAISS, Neo4j, or specific LLM providers.

Usage:
    service = SearchService(vector_store_factory, graph_store, llm_provider)
    docs = service.retrieve("what is the attention mechanism?")
"""
from langchain_core.documents import Document

from Src.Config.settings import Settings
from Src.Interfaces.IGraphStore import IGraphStore
from Src.Interfaces.ILLMProvider import ILLMProvider
from Src.Interfaces.IVectorStore import IVectorStore


class SearchService:
    """
    Multi-source retrieval service — combines vector similarity search
    with graph database querying.

    The dynamic index selection (choosing the best FAISS index per query)
    is handled here using the LLM provider for routing.
    """

    def __init__(
        self,
        vector_store_factory: "VectorStoreFactory",
        graph_store: IGraphStore,
        llm_provider: ILLMProvider,
    ) -> None:
        self._vector_store_factory = vector_store_factory
        self._graph_store = graph_store
        self._llm_provider = llm_provider

    def retrieve(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """
        Retrieve relevant documents from all configured sources.

        Args:
            query: Natural language query.
            k: Max results from the vector store.

        Returns:
            Combined list of (Document, float) tuples, graph results scored at 0.0.
        """
        results: list[tuple[Document, float]] = []

        # Graph first — high-confidence structured knowledge
        graph_answer = self._graph_store.query(query)
        if graph_answer:
            results.append(
                (Document(page_content=f"[GRAPH KNOWLEDGE]: {graph_answer}"), 0.0)
            )

        # Vector search — semantic similarity
        vector_results = self._vector_store_factory.search_best(query, k=k)
        results.extend(vector_results)

        return results


class VectorStoreFactory:
    """
    Manages multiple named FAISS indexes and routes queries to the best one.

    This replaces the old select_best_index() logic from faiss_search.py,
    keeping it properly encapsulated.
    """

    def __init__(self, settings: Settings, llm_provider: ILLMProvider) -> None:
        self._settings = settings
        self._llm_provider = llm_provider

    def get_store(self, index_name: str) -> IVectorStore:
        """Get or create a vector store for a specific index name."""
        from Src.Providers.FAISSVectorStore import FAISSVectorStore
        return FAISSVectorStore(self._settings, index_name)

    def list_indexes(self) -> list[str]:
        """Return names of all available FAISS indexes."""
        import os
        base = self._settings.vector_store_base_path
        if not base.exists():
            return []
        return [d for d in os.listdir(base) if (base / d).is_dir()]

    def search_best(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """
        Route the query to the most relevant FAISS index and run search.
        Uses LLM routing for multi-index disambiguation.
        """
        indexes = self.list_indexes()
        if not indexes:
            return []

        best_index = self._select_index(query, indexes)
        store = self.get_store(best_index)
        try:
            return store.search(query, k=k)
        except Exception as exc:
            print(f"[VectorStoreFactory] Search failed on index '{best_index}': {exc}")
            return []

    def _select_index(self, query: str, indexes: list[str]) -> str:
        """Use an LLM to pick the best index, or return the first if only one exists."""
        if len(indexes) == 1:
            return indexes[0]

        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field

        class IndexDecision(BaseModel):
            index_name: str = Field(description="The exact name of the best index.")

        try:
            llm = self._llm_provider.get_llm(performance="standard")
            structured = llm.with_structured_output(IndexDecision)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Choose the best index for the query. Available: {indexes}"),
                ("human", "{query}"),
            ])
            result = (prompt | structured).invoke(
                {"query": query, "indexes": ", ".join(indexes)}
            )
            if result and result.index_name in indexes:
                return result.index_name
        except Exception:
            pass

        return indexes[0]
