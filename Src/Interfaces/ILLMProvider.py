"""
ILLMProvider — Abstract contract for any LLM backend.

Plug-and-Play: To swap Groq for OpenAI, Anthropic, or Ollama,
create a new class that implements this interface. Only container.py needs to change.
"""
from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel


class ILLMProvider(ABC):
    """
    Abstract base for all LLM backend providers.
    Implementations: GroqLLMProvider, OpenAILLMProvider, OllamaLLMProvider, etc.

    The 'performance' tier allows callers to request different model sizes
    without knowing anything about the underlying provider.

    Performance tiers:
        "standard" — Faster, cheaper model (good for routing, grading, intent detection)
        "high"     — More capable model (good for graph extraction, complex reasoning)
    """

    @abstractmethod
    def get_llm(self, performance: str = "standard") -> BaseChatModel:
        """
        Return a ready-to-use LangChain chat model.

        Args:
            performance: "standard" or "high". Provider chooses the appropriate model.

        Returns:
            A LangChain-compatible BaseChatModel instance.
        """
        ...
