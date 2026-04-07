"""
Centralized application configuration using Pydantic BaseSettings.
All environment variables are loaded here — no more scattered os.getenv() calls.

Usage:
    from Src.Config.settings import settings
    print(settings.groq_api_key)
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is 2 levels up from this file: Src/Config/ → Src/ → Root
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    All application configuration in one place.
    Values are read from the .env file at the project root.
    Adding a new setting = add one field here. Nothing else changes.
    """

    # --- LLM / Groq ---
    groq_api_key: str = "your_groq_api_key_here"
    groq_model_name: str = "openai/gpt-oss-120b"

    # --- Neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""

    # --- Embedding ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage paths (derived, not from .env) ---
    vector_store_base_path: Path = _ROOT_DIR / "Src" / "Vector" / "faiss"
    uploads_path: Path = _ROOT_DIR / "Src" / "Uploads"

    # --- Model safety list ---
    # Models known to be decommissioned — never use these
    decommissioned_models: list[str] = ["llama3-70b-8192", "llama-3.1-8b-instant"]

    # --- Pydantic config ---
    model_config = SettingsConfigDict(
        env_file=str(_ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",        # ignore unknown .env keys
    )

    def is_groq_available(self) -> bool:
        """Check if a valid Groq API key is configured."""
        return bool(self.groq_api_key) and self.groq_api_key != "your_groq_api_key_here"

    def is_model_safe(self, model_name: str) -> bool:
        """Check if a model name is not in the decommissioned list."""
        return not any(bad in model_name.lower() for bad in self.decommissioned_models)


# Singleton instance — import this everywhere
settings = Settings()
