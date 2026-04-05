from Src.Evaluaters.Hellucination_Grader import result_evaluater

from .Agent_state import AgentState


def result_evaluator(state: AgentState):
    response = state["response"]
    query = state["query"]
    intent = state.get("query_evaluator", "search")
    
    if not isinstance(response, str):
        response = str(response) if response is not None else ""
        
    # Greetings shouldn't undergo rigorous hallucination grading
    if intent == "greeting":
        return {"response": response, "response_checker": "yes"}
        
    verdict = result_evaluater(query, response)
    return {
        "response": response,
        "response_checker": "yes" if verdict == "yes" else "no",
    }
