# Dynamic Hybrid GraphRAG Agent 🔮

A robust, fully portable Retrieval-Augmented Generation (RAG) backend utilizing **LangGraph**, **FastAPI**, **FAISS**, and **Neo4j**. 

This system represents a next-generational "Hybrid RAG" architecture. By combining standard semantic similarity (FAISS) with high-fidelity factual relationships mapped as graph nodes and edges (Neo4j GraphRAG), the chatbot is capable of answering complex multi-hop reasoning questions cleanly.

## Key Features
- **Dynamic Vector Databases**: Instead of a global swamp of chunks, each uploaded PDF gets its very own FAISS index dynamically named after the file. A smart router selects the relevant FAISS repository to search.
- **Neo4j Graph Integration**: Uses `LLMGraphTransformer` to chew through text, extract core Entities and Relationships, and autonomously build a Knowledge Graph in your local/remote Neo4j instance during the embedding phase.
- **Hybrid Retrieval Strategy**: The retriever node searches your vector index AND automatically parses the user's natural language into a Cypher query to pull graph data. Both contexts are merged.
- **Memory & Cross-Turning Context**: Leverages LangGraph `MemorySaver` so you can ask follow-up questions organically.
- **Intent Filtration**: A dedicated routing node sits at the graph's `START`. If a user initiates a harmless greeting like "Hello", the system skips expensive vector operations entirely and routes straight to a friendly chatbot response.
- **Portability**: Agnostic `.env` injection allows rapid model hot-swapping between `Ollama` and lightning-fast `Groq` inference models.

---

## 🚀 Quickstart Guide

### 1. Requirements

Make sure you have [Python 3.10+](https://www.python.org/downloads/) installed. Recommended to use `uv` or `venv` for virtual environments.

```bash
# Clone the repository
git clone https://github.com/your-username/Dynamic-GraphRAG.git
cd Dynamic-GraphRAG

# Install all dependencies
pip install -r requirement.txt
```

### 2. Environment Variables

Create a `.env` file in the root codebase and insert your Neo4j credentials and LLM inferencing keys:

```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=gpt/openai-oss-120b

NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Usage

Start the blazing fast Uvicorn ASGI server!

```bash
uvicorn main:app --reload
```

#### Endpoints
- `POST /upload`: Securely upload PDFs into the local workspace.
- `GET /embed/{file_name}`: Extract chunks, evaluate entities, generate a dynamic FAISS store and insert knowledge graphs into Neo4j.
- `GET /chat?query=Your%20Question`: Hit the primary LangGraph agent timeline with your questions.

*(Alternatively, use `python test_invoke.py` from the command line to debug the LangGraph architecture without the HTTP server wrapper.)*

---

## 🛠️ Project Structure
- `Src/Agents/`: The individual nodes for mapping out the precise LangGraph execution workflow.
- `Src/Router/`: The FastAPI routers exposing endpoints functionally.
- `Src/Embeddings/`: Handles text splitting and pushing data into FAISS + Neo4j. 
- `Src/Evaluaters/`: Houses prompt boundaries for grading hallucination, correcting queries, and scoring knowledge retrieval.
- `VectoreStore/`: Interfaces with the local FAISS directories and manages dynamic indexing loads.
- `test_invoke.py`: A local python script designed to simulate a user request to LangGraph dynamically tracing error origins.
