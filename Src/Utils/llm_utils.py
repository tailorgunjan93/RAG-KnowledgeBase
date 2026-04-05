import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env is loaded from the project root
current_dir = Path(__file__).resolve().parent
# Src/Utils/llm_utils.py -> Src/Utils/ -> Src/ -> Root
root_dir = current_dir.parent.parent.parent
env_path = root_dir / '.env'

# Use absolute path to ensure .env is found in all execution contexts
load_dotenv(dotenv_path=str(env_path.resolve()))

def setup_neo4j():
    """Sets Neo4j environment variables so Neo4jGraph() can find them automatically."""
    os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI", "")
    os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME", "")
    os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD", "")

def get_llm(performance="standard"):
    """
    Centralized LLM Factory. 
    'performance' can be 'standard' (faster/cheaper) or 'high' (more accurate/extraction).
    """
    api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL_NAME")
    
    # Define currently supported/verified model names for Groq
    HIGH_PERF_MODEL = "openai/gpt-oss-120b"
    STANDARD_MODEL = "openai/gpt-oss-120b"
    
    # List of known decommissioned/broken models to avoid
    DECOMMISSIONED = ["llama3-70b-8192", "llama-3.1-8b-instant"]
    
    if not api_key or api_key == "your_groq_api_key_here":
        # Fallback to Ollama if no Groq API Key
        model = "llama3.2" if performance == "standard" else "llama3.2"
        return ChatOllama(model=model)
    
    # If the user has a model set in .env, check if it's safe to use
    is_safe = groq_model and not any(bad in groq_model.lower() for bad in DECOMMISSIONED)
    
    if performance == "high":
        model = groq_model if is_safe else HIGH_PERF_MODEL
    else:
        # For standard tasks, default to a fast model unless a safe small model is provided
        model = groq_model if (is_safe and "70b" not in groq_model.lower()) else STANDARD_MODEL
        
    return ChatGroq(model=model, api_key=api_key)
