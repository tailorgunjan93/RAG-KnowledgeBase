from langchain_community.vectorstores import FAISS
import Src.Embeddings.embeddings as embeddings
from pathlib import Path
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

base_vectorStore_path = embeddings.base_vectorStore_path

class RouteDecision(BaseModel):
    index_name: str = Field(description="The exact name of the index folder to search.")

def get_available_indexes():
    if not os.path.exists(base_vectorStore_path):
        return []
    return [d for d in os.listdir(base_vectorStore_path) if os.path.isdir(os.path.join(base_vectorStore_path, d))]

def select_best_index(query: str):
    indexes = get_available_indexes()
    if not indexes:
        return None
    if len(indexes) == 1:
        return indexes[0]
        
    from Src.Utils.llm_utils import get_llm
    llm = get_llm(performance="standard")
        
    structured_llm = llm.with_structured_output(RouteDecision)
    system = "You are an index routing assistant. Choose the best matching document index to answer the user query. Available indexes: {indexes}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{query}")
    ])
    try:
        decision = (prompt | structured_llm).invoke({"query": query, "indexes": ", ".join(indexes)})
        if decision and decision.index_name in indexes:
            return decision.index_name
    except Exception as e:
        print(f"Error in select_best_index: {e}")
    
    return indexes[0] # fallback

def search_dynamic_faiss_index_with_score(query: str, k: int = 4):
    best_index = select_best_index(query)
    if not best_index:
        return []
        
    target_path = base_vectorStore_path / best_index
    try:
        index = embeddings.load_faiss_index(str(target_path), embeddings.embedding)
        results = index.similarity_search_with_score(query, k=k)
        # Fix LangGraph serialization issue by converting numpy.float32 scores to Python floats
        return [(doc, float(score)) for doc, score in results]
    except Exception as e:
        print(f"Error loading faiss index {best_index}: {e}")
        return []

def search_neo4j_graph(query: str):
    try:
        from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
        from Src.Utils.llm_utils import get_llm, setup_neo4j
        
        # Centralized setup for Neo4j credentials
        setup_neo4j()
        
        llm = get_llm(performance="standard")
            
        graph = Neo4jGraph()
        chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, return_direct=True, allow_dangerous_requests=True)
        result = chain.invoke({"query": query})
        
        from langchain_core.documents import Document
        # Wrap neo4j answer into a document context so the pipeline handles it easily
        if "result" in result and result["result"]:
            # give it a perfect score (0.0) so it's prioritized
            return [(Document(page_content="[GRAPH KNOWLEDGE]: " + str(result["result"])), 0.0)]
    except Exception as e:
        print(f"Neo4j graph query failed: {e}")
    return []
