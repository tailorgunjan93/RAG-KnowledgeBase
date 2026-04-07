"""
llm_utils.py — Backward-compatible LLM accessor.

IMPORTANT: This module now delegates to the centralized DI container.
All model-selection logic has moved to Src/Providers/GroqLLMProvider.py.
All configuration has moved to Src/Config/settings.py.

Existing code that calls get_llm() and setup_neo4j() continues to work unchanged.
This is the ONLY file that bridges old and new architecture during transition.
"""
from Src.Config.settings import settings as _settings


def get_llm(performance: str = "standard"):
    """
    Get a LangChain chat model for the requested performance tier.

    Delegates to GroqLLMProvider — all model-selection logic is there.

    Args:
        performance: "standard" (fast/cheap) or "high" (capable/extraction).

    Returns:
        A LangChain-compatible BaseChatModel instance.
    """
    from Src.Providers.GroqLLMProvider import GroqLLMProvider
    return GroqLLMProvider(_settings).get_llm(performance=performance)


def setup_neo4j() -> None:
    """
    Ensure Neo4j credentials are in os.environ so Neo4jGraph() can find them.

    Delegates to Neo4jGraphStore._inject_neo4j_env() logic using centralized settings.
    Kept here for backward compatibility with older call sites.
    """
    import os
    os.environ["NEO4J_URI"] = _settings.neo4j_uri
    os.environ["NEO4J_USERNAME"] = _settings.neo4j_username
    os.environ["NEO4J_PASSWORD"] = _settings.neo4j_password
