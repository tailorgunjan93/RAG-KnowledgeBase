"""
IGraphStore — Abstract contract for any knowledge graph database.

Plug-and-Play: To swap Neo4j for Amazon Neptune or a local graph store,
create a new class that implements this interface. Only container.py needs to change.
"""
from abc import ABC, abstractmethod
from langchain_core.documents import Document


class IGraphStore(ABC):
    """
    Abstract base for all graph database providers.
    Implementations: Neo4jGraphStore, InMemoryGraphStore, etc.
    """

    @abstractmethod
    def add_graph_documents(self, docs: list[Document]) -> None:
        """
        Extract entities & relationships from documents and write to the graph.

        Args:
            docs: LangChain Document objects to process with an LLM graph transformer.
        """
        ...

    @abstractmethod
    def query(self, question: str) -> str:
        """
        Run a natural-language graph query (Cypher generation internally).

        Args:
            question: User's natural language question.

        Returns:
            A plain text answer derived from the graph, or an empty string if no result.
        """
        ...
