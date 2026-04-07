"""
EmbeddingService — Orchestrates the full document ingestion pipeline.

SRP: This service ONLY coordinates loading → vector storage → graph storage.
It does NOT know which vector DB or which LLM is being used — that's the
providers' concern. Full dependency inversion.

Usage:
    service = EmbeddingService(loader, vector_store_factory, graph_store)
    count = await service.process_file("/path/to/document.pdf")
"""
from pathlib import Path

from Src.Interfaces.IDocumentLoader import IDocumentLoader
from Src.Interfaces.IGraphStore import IGraphStore
from Src.Interfaces.IVectorStore import IVectorStore


class EmbeddingService:
    """
    Orchestrates the document ingestion pipeline:
      1. Load & chunk the document (via IDocumentLoader)
      2. Embed & store in vector DB (via IVectorStore)
      3. Extract entities & store in graph DB (via IGraphStore)

    All three steps are injected — swap any backend without touching this class.
    """

    def __init__(
        self,
        loader: IDocumentLoader,
        vector_store: IVectorStore,
        graph_store: IGraphStore,
    ) -> None:
        self._loader = loader
        self._vector_store = vector_store
        self._graph_store = graph_store

    async def process_file(self, file_path: str | Path) -> int:
        """
        Full ingestion pipeline for a single file.

        Args:
            file_path: Absolute path to the source document.

        Returns:
            Number of vector chunks stored.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        print(f"[EmbeddingService] Processing: {file_path.name}")

        # Step 1: Load and chunk
        docs = await self._loader.load(str(file_path))
        print(f"[EmbeddingService] Loaded {len(docs)} chunks.")

        # Step 2: Vector store
        self._vector_store.add_documents(docs)
        count = self._vector_store.document_count
        print(f"[EmbeddingService] Vector store now has {count} vectors.")

        # Step 3: Graph store (non-blocking — failure here won't break the pipeline)
        try:
            self._graph_store.add_graph_documents(docs)
        except Exception as exc:
            print(f"[EmbeddingService] Graph store failed (non-fatal): {exc}")

        return count
