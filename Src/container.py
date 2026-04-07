"""
container.py — Dependency Injection Container (the "plug board").

THIS is the only place where concrete implementations are wired to interfaces.
To swap any backend, change ONE line here. Nothing else in the codebase changes.

Examples of future swaps:
    # Swap vector DB:      vector_store_factory = ChromaVectorStoreFactory(settings)
    # Swap LLM provider:   llm_provider = OpenAILLMProvider(settings)
    # Swap document loader: loader = UnstructuredDocumentLoader()
    # Swap graph DB:       graph_store = NeptuneGraphStore(settings, llm_provider)
"""
from Src.Config.settings import settings

# ── LLM Provider ─────────────────────────────────────────────────────────────
# To use OpenAI instead: from Src.Providers.OpenAILLMProvider import OpenAILLMProvider
from Src.Providers.GroqLLMProvider import GroqLLMProvider

llm_provider = GroqLLMProvider(settings)

# ── Graph Store ──────────────────────────────────────────────────────────────
# To use Neptune instead: from Src.Providers.NeptuneGraphStore import NeptuneGraphStore
from Src.Providers.Neo4jGraphStore import Neo4jGraphStore

graph_store = Neo4jGraphStore(settings, llm_provider)

# ── Vector Store Factory ─────────────────────────────────────────────────────
# To use Chroma instead: build a ChromaVectorStoreFactory(settings)
from Src.Services.SearchService import VectorStoreFactory

vector_store_factory = VectorStoreFactory(settings, llm_provider)

# ── Document Loader ──────────────────────────────────────────────────────────
# To support DOCX instead: from Src.Providers.DocxDocumentLoader import DocxDocumentLoader
from Src.Providers.PyPDFDocumentLoader import PyPDFDocumentLoader

loader = PyPDFDocumentLoader()

# ── Services ─────────────────────────────────────────────────────────────────
from Src.Services.EmbeddingService import EmbeddingService
from Src.Services.SearchService import SearchService
from Src.Services.ChatService import ChatService


def make_embedding_service(index_name: str) -> EmbeddingService:
    """
    Create an EmbeddingService pre-wired to the correct vector store index.
    Called per-request since each file has its own index name.
    """
    vector_store = vector_store_factory.get_store(index_name)
    return EmbeddingService(loader, vector_store, graph_store)


search_service = SearchService(vector_store_factory, graph_store, llm_provider)
chat_service = ChatService()
