from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain_community.vectorstores import FAISS
from Src.Ingestion.ingest import ingest_file
import os

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
base_vectorStore_path = Path(__file__).parent.parent / "Vector" / "faiss"

async def embed_file(file_path: str):
    docs = await ingest_file(file_path)
    file_stem = Path(file_path).stem
    dynamic_vectorStore_path = base_vectorStore_path / file_stem
    
    if not os.path.exists(dynamic_vectorStore_path):       
        faiss_index = await create_faiss_index(docs, str(dynamic_vectorStore_path), embedding)
        faiss_index = await save_faiss_index(faiss_index, str(dynamic_vectorStore_path))
    else:
        faiss_index = load_faiss_index(str(dynamic_vectorStore_path), embedding)
        faiss_index = await add_documents(faiss_index, docs)
        faiss_index = await save_faiss_index(faiss_index, str(dynamic_vectorStore_path))
        
    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
        from langchain_neo4j import Neo4jGraph
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model="llama3.2")
        else:
            from langchain_groq import ChatGroq
            model_name = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
            llm = ChatGroq(model=model_name, api_key=api_key)
            
        llm_transformer = LLMGraphTransformer(llm=llm)
        graph_documents = llm_transformer.convert_to_graph_documents(docs)
        
        graph = Neo4jGraph() # Reads from NEO4J_URI environment variables automatically
        graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
        print(f"Added graph docs to Neo4j for {file_stem}")
    except Exception as e:
        print(f"Skipped Neo4j graph generation: {e}")

    return faiss_index

async def create_faiss_index(docs, vectorStore_path: str, embeddings):
    faiss_index = FAISS.from_documents(docs, embedding=embeddings)
    return faiss_index

def load_faiss_index(vectorStore_path: str, embeddings):
    faiss_index = FAISS.load_local(folder_path=vectorStore_path, embeddings=embeddings, index_name="index", allow_dangerous_deserialization=True)
    return faiss_index

async def add_documents(faiss_index: FAISS, docs):
    faiss_index.add_documents(docs)
    return faiss_index

async def save_faiss_index(faiss_index: FAISS, vectorStore_path: str):
    faiss_index.save_local(vectorStore_path)
    return faiss_index