from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os

from Src.Evaluaters.QueryGrader import QueryGrader
from .Agent_state import AgentState
from Src.Utils.llm_utils import get_llm

class RouteDecision(BaseModel):
    route: str = Field(description="Exactly 'web_search' or 'chat_bot'.")

def query_evaluation(state: AgentState):
    document = state.get("document", [])
    query = state["query"]
    
    # If we have documents, let's grade them.
    if document:
        is_relevant = False
        for item in document:
            if isinstance(item, tuple):
                doc = item[0]
                doc_content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            elif hasattr(item, "page_content"):
                doc_content = item.page_content
            else:
                doc_content = str(item)

            if QueryGrader(query, doc_content) == "yes":
                is_relevant = True
                break
        
        if is_relevant:
            return {"query_evaluator": "chat_bot"}

    # If no documents or they are irrelevant, decide if we need web search
    llm = get_llm(performance="standard")
    
    llm_structured = llm.with_structured_output(RouteDecision)
    
    system = """You are a routing expert for an AI agent. 
    The system either found no relevant internal documents or the user is just chatting.
    
    Determine if the user's query requires an internet search (searching for facts, news, or external data) or if the AI should just respond directly (conversational chat, personal questions, or simple statements).
    
    Return 'web_search' ONLY if the user is asking for specific information that requires external research.
    Return 'chat_bot' if it's a personal statement, a conversational follow-up, or something the AI can answer as a helpful assistant without searching the web.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", f"User query: {query}")
    ])
    
    try:
        decision = (prompt | llm_structured).invoke({})
        return {"query_evaluator": decision.route if decision else "chat_bot"}
    except Exception as e:
        print(f"Routing error: {e}")
        return {"query_evaluator": "chat_bot"}
