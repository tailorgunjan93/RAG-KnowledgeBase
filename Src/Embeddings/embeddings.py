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
        
        from Src.Utils.llm_utils import get_llm, setup_neo4j
        
        # Centralized setup for Neo4j credentials
        setup_neo4j()
        
        print(f"--- Starting Graph Extraction for {file_stem} ---")
        
        from Src.Utils.llm_utils import get_llm
        # Using a high-performance LLM for accurate graph document extraction
        llm = get_llm(performance="high")
        
        print(f"Using LLM for Extraction: {llm.model_name if hasattr(llm, 'model_name') else 'Ollama'}")
        llm_transformer = LLMGraphTransformer(llm=llm)
        
        print("Converting documents to graph documents (this can take some time)...")
        graph_documents = llm_transformer.convert_to_graph_documents(docs)
        print(f"Extracted {len(graph_documents)} graph documents.")
        
        if len(graph_documents) > 0:
            print(f"Connecting to Neo4j at {os.environ['NEO4J_URI']}...")
            graph = Neo4jGraph() # Will now read from os.environ
            
            print("Writing to Neo4j...")
            graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
            print(f"✅ SUCCESS: Added graph documents to Neo4j for {file_stem}")
        else:
            print("⚠️ WARNING: No graph documents were extracted from this file.")
        
    except Exception as e:
        import traceback
        print(f"❌ FAILED Neo4j graph generation: {e}")
        traceback.print_exc()

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