from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .Agent_state import state
from .chat_bot_node import chat_bot
from .query_corrector_node import query_correction
from .query_evaluation_node import query_evaluation
from .result_evaluator_node import result_evaluator
from .retriever_node import retriever
from .web_search_node import web_search
from .intent_detector_node import intent_detector

#adding graph workflow 
graph_workflow = StateGraph(state)

#Adding nodes in langgraph
graph_workflow.add_node("intent_detector", intent_detector)
graph_workflow.add_node("retriever",retriever)
graph_workflow.add_node("chat_bot",chat_bot)
graph_workflow.add_node("web_search",web_search)
graph_workflow.add_node("query_correction",query_correction)
graph_workflow.add_node("result_evaluator",result_evaluator)
graph_workflow.add_node("query_evaluation",query_evaluation)

def intent_decider(state: state):
    # if greeting, go to chatbot directly bypassing retrieval
    return "chat_bot" if state.get("query_evaluator") == "greeting" else "retriever"

def retrieve_eval_decider(state: state):
    return state["query_evaluator"]

def result_eval_decide(state: state):
    return "END" if state.get("response_checker") == "yes" else "query_correction"

#adding edges
graph_workflow.add_edge(START,"intent_detector")
graph_workflow.add_conditional_edges(
    "intent_detector",
    intent_decider,
    {
        "chat_bot": "chat_bot",
        "retriever": "retriever"
    }
)

graph_workflow.add_edge("retriever","query_evaluation")
graph_workflow.add_conditional_edges(
    "query_evaluation",
    retrieve_eval_decider,
    {
        "web_search":"web_search",
        "chat_bot":"chat_bot"
    }
)
graph_workflow.add_edge("web_search","result_evaluator")
graph_workflow.add_edge("chat_bot","result_evaluator")
graph_workflow.add_conditional_edges(
    "result_evaluator",
    result_eval_decide,
    {
        "END":END,
        "query_correction":"query_correction"
    }
)
graph_workflow.add_edge("query_correction","retriever")

memory = MemorySaver()
app = graph_workflow.compile(checkpointer=memory)
