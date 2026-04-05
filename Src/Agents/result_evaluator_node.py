from Src.Evaluaters.Hellucination_Grader import result_evaluater

from .Agent_state import state


def result_evaluator(state: state):
    response = state["response"]
    query = state["query"]
    if not isinstance(response, str):
        response = str(response) if response is not None else ""
    verdict = result_evaluater(query, response)
    return {
        "response": response,
        "response_checker": "yes" if verdict == "yes" else "no",
    }
