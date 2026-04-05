from VectoreStore.faiss_search import search_dynamic_faiss_index_with_score, search_neo4j_graph
from .Agent_state import AgentState


def retriever(state: AgentState):
    query = state["query"]
    
    # We do a sequential or parallel execution to get both Graph and Vector results.
    neo4j_results = search_neo4j_graph(query=query)
    faiss_results = search_dynamic_faiss_index_with_score(query=query)
    
    # Combine the contextual results from Graph and Vector DBs
    docs = neo4j_results + faiss_results
    
    return {"document": docs}
