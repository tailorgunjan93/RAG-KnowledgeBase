"""
Neo4jGraphStore — Concrete implementation of IGraphStore for Neo4j.

Wraps all Neo4j logic from embeddings.py and faiss_search.py into one class.
Handles LLM graph extraction (add_graph_documents) and Cypher QA (query).

To swap for Amazon Neptune: create NeptuneGraphStore(IGraphStore) and update container.py.
"""
from langchain_core.documents import Document

from Src.Config.settings import Settings
from Src.Interfaces.IGraphStore import IGraphStore
from Src.Interfaces.ILLMProvider import ILLMProvider


class Neo4jGraphStore(IGraphStore):
    """
    Manages entity/relationship extraction into Neo4j and natural-language querying.

    Two responsibilities (both scoped to graph data — Single Responsibility OK):
      1. add_graph_documents: LLM → entity extraction → Neo4j write
      2. query: Natural language → Cypher → Neo4j read → plain text answer
    """

    def __init__(self, settings: Settings, llm_provider: ILLMProvider) -> None:
        """
        Args:
            settings: Application settings (Neo4j credentials).
            llm_provider: LLM provider for entity extraction and Cypher generation.
        """
        self._settings = settings
        self._llm_provider = llm_provider

    # ── IGraphStore implementation ───────────────────────────────────────────

    def add_graph_documents(self, docs: list[Document]) -> None:
        """
        Extract entities & relationships from documents and write to Neo4j.

        Uses a high-performance LLM for accurate graph extraction.
        Silently skips if Neo4j is not configured or extraction yields nothing.
        """
        try:
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            from langchain_neo4j import Neo4jGraph

            self._inject_neo4j_env()

            llm = self._llm_provider.get_llm(performance="high")
            print(f"[Neo4jGraphStore] Using LLM: {getattr(llm, 'model_name', 'Ollama')}")

            transformer = LLMGraphTransformer(llm=llm)
            print(f"[Neo4jGraphStore] Extracting graph from {len(docs)} documents...")
            graph_docs = transformer.convert_to_graph_documents(docs)
            print(f"[Neo4jGraphStore] Extracted {len(graph_docs)} graph documents.")

            if not graph_docs:
                print("[Neo4jGraphStore] ⚠️ No graph documents extracted — skipping Neo4j write.")
                return

            graph = Neo4jGraph()
            graph.add_graph_documents(
                graph_docs,
                baseEntityLabel=True,
                include_source=True,
            )
            print(f"[Neo4jGraphStore] ✅ Successfully wrote to Neo4j.")

        except Exception as exc:
            import traceback
            print(f"[Neo4jGraphStore] ❌ Graph extraction failed: {exc}")
            traceback.print_exc()

    def query(self, question: str) -> str:
        """
        Run a natural-language question against the Neo4j graph.

        Returns:
            Plain text answer, or empty string on failure/no result.
        """
        try:
            from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

            self._inject_neo4j_env()
            llm = self._llm_provider.get_llm(performance="standard")

            graph = Neo4jGraph()
            chain = GraphCypherQAChain.from_llm(
                graph=graph,
                llm=llm,
                verbose=True,
                return_direct=True,
                allow_dangerous_requests=True,
            )
            result = chain.invoke({"query": question})
            answer = result.get("result", "")
            return str(answer) if answer else ""

        except Exception as exc:
            print(f"[Neo4jGraphStore] Graph query failed: {exc}")
            return ""

    # ── Private helpers ──────────────────────────────────────────────────────

    def _inject_neo4j_env(self) -> None:
        """
        Ensure Neo4j credentials are in os.environ so Neo4jGraph() can find them.
        Neo4jGraph reads from env vars directly — this is a library constraint.
        """
        import os
        os.environ["NEO4J_URI"] = self._settings.neo4j_uri
        os.environ["NEO4J_USERNAME"] = self._settings.neo4j_username
        os.environ["NEO4J_PASSWORD"] = self._settings.neo4j_password
