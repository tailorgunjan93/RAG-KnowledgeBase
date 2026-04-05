from Src.Evaluaters.QueryCorrector import query_corrector

from .Agent_state import AgentState

def query_correction(state: AgentState):
    query = state["query"]
    response = state["response"]
    new_query = query_corrector(query, response)
    return {"query": new_query}
