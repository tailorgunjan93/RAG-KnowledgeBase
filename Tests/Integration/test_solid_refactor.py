"""Smoke test for SOLID refactoring - Phase 1-6 validation."""
import sys

def test_settings():
    from Src.Config.settings import settings
    assert settings.groq_model_name, "groq_model_name must be set"
    print(f"[OK] Settings - Model: {settings.groq_model_name}")
    print(f"     Neo4j URI: {settings.neo4j_uri}")
    return settings

def test_interfaces():
    from Src.Interfaces.ILLMProvider import ILLMProvider
    from Src.Interfaces.IVectorStore import IVectorStore
    from Src.Interfaces.IGraphStore import IGraphStore
    from Src.Interfaces.IDocumentLoader import IDocumentLoader
    print("[OK] All 4 Interfaces imported")

def test_llm_provider(settings):
    from Src.Providers.GroqLLMProvider import GroqLLMProvider
    provider = GroqLLMProvider(settings)
    llm_std = provider.get_llm("standard")
    llm_high = provider.get_llm("high")
    model_name = getattr(llm_std, "model_name", "ollama")
    print(f"[OK] GroqLLMProvider - standard: {model_name}")
    model_name_high = getattr(llm_high, "model_name", "ollama")
    print(f"[OK] GroqLLMProvider - high: {model_name_high}")

def test_llm_utils_bridge():
    from Src.Utils.llm_utils import get_llm, setup_neo4j
    llm = get_llm("standard")
    model_name = getattr(llm, "model_name", "ollama")
    print(f"[OK] llm_utils bridge - model: {model_name}")
    setup_neo4j()
    print("[OK] setup_neo4j() runs without error")

def test_faiss_provider(settings):
    from Src.Providers.FAISSVectorStore import FAISSVectorStore
    store = FAISSVectorStore(settings, "smoke_test_dont_save")
    print("[OK] FAISSVectorStore instantiated")

def test_services(settings):
    from Src.Services.ChatService import ChatService
    chat = ChatService()
    print("[OK] ChatService instantiated")

    from Src.Services.SearchService import VectorStoreFactory, SearchService
    from Src.Providers.GroqLLMProvider import GroqLLMProvider
    from Src.Providers.Neo4jGraphStore import Neo4jGraphStore
    factory = VectorStoreFactory(settings, GroqLLMProvider(settings))
    indexes = factory.list_indexes()
    print(f"[OK] VectorStoreFactory - found {len(indexes)} existing indexes: {indexes}")

def test_container():
    from Src.container import (
        settings, llm_provider, graph_store,
        vector_store_factory, loader, search_service,
        chat_service, make_embedding_service
    )
    print("[OK] Container wired successfully - all singletons created")

if __name__ == "__main__":
    errors = []
    steps = [
        ("Settings", lambda: test_settings()),
        ("Interfaces", lambda: test_interfaces()),
    ]

    settings = None
    try:
        settings = test_settings()
    except Exception as e:
        print(f"[FAIL] Settings: {e}")
        sys.exit(1)

    for name, fn in [
        ("Interfaces", test_interfaces),
        ("LLM Provider", lambda: test_llm_provider(settings)),
        ("llm_utils bridge", test_llm_utils_bridge),
        ("FAISS Provider", lambda: test_faiss_provider(settings)),
        ("Services", lambda: test_services(settings)),
        ("Container", test_container),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            errors.append(name)

    print()
    if errors:
        print(f"FAILED: {errors}")
        sys.exit(1)
    else:
        print("ALL PHASES 1-6 VERIFIED SUCCESSFULLY")
