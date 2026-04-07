"""
GroqLLMProvider — Concrete implementation of ILLMProvider for Groq + Ollama.

This class centralizes ALL model-selection logic.
To switch to OpenAI: create OpenAILLMProvider(ILLMProvider) and update container.py.
"""
from langchain_core.language_models.chat_models import BaseChatModel

from Src.Config.settings import Settings
from Src.Interfaces.ILLMProvider import ILLMProvider


class GroqLLMProvider(ILLMProvider):
    """
    Provides LangChain chat models backed by Groq Cloud API.
    Falls back to Ollama (local) if no valid API key is configured.

    Model Selection Strategy:
        - "high" performance → uses groq_model_name from settings (default: 120b)
        - "standard" performance → uses a fast/cheap model (avoids 70b heavyweights)
        - If model_name is decommissioned → falls back to verified safe defaults
    """

    # Verified, currently-running fallback defaults
    _HIGH_PERF_FALLBACK = "openai/gpt-oss-120b"
    _STANDARD_FALLBACK = "openai/gpt-oss-120b"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_llm(self, performance: str = "standard") -> BaseChatModel:
        """
        Return a ready-to-use LangChain chat model for the requested performance tier.

        Args:
            performance: "standard" (fast/cheap) or "high" (capable/extraction).

        Returns:
            ChatGroq or ChatOllama instance.
        """
        if not self._settings.is_groq_available():
            return self._get_ollama_llm()

        model = self._resolve_model(performance)
        return self._get_groq_llm(model)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resolve_model(self, performance: str) -> str:
        """Pick the best model name based on performance tier and safety checks."""
        configured = self._settings.groq_model_name
        is_safe = self._settings.is_model_safe(configured)

        if performance == "high":
            return configured if is_safe else self._HIGH_PERF_FALLBACK
        else:
            # Avoid sending a 70b heavyweight to a "standard" (fast) slot
            is_heavy = "70b" in configured.lower() and not is_safe
            if is_safe and not is_heavy:
                return configured
            return self._STANDARD_FALLBACK

    def _get_groq_llm(self, model: str) -> BaseChatModel:
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=self._settings.groq_api_key)

    def _get_ollama_llm(self) -> BaseChatModel:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2")
