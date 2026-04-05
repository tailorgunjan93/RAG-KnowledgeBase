from langchain_groq import ChatGroq
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from .Agent_state import AgentState

# Using a faster/cheaper model for intent detection
class IntentClassification(BaseModel):
    intent: str = Field(description="Exactly 'greeting' or 'search', indicating if the user query is just saying hello/greeting, or actually searching/asking a question.")

def intent_detector(state: AgentState):
    query = state["query"].strip().lower()
    
    # Heuristic check for common greetings and chatting (instant)
    common_chat = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you", "who are you", "i am", "my name is", "nice to meet you"}
    if any(greet in query for greet in common_chat) and len(query) < 40: # Extended limit to catch 'My name is Gunjan Tailor'
        return {"query_evaluator": "greeting", "messages": [HumanMessage(content=state["query"])]}

    try:
        # Using the centralized LLM factory
        from Src.Utils.llm_utils import get_llm
        llm = get_llm(performance="standard")
            
        llm_structured = llm.with_structured_output(IntentClassification)
        
        system = """You are an intent classifier for a knowledge base agent.
        Determine if the user's message is a greeting (like 'hello'), personal introduction (like 'I am Gunjan'), or casual chatting (like 'how can you help me') OR if it's a specific question/search query for information.
        
        Return 'greeting' for: greetings, introductions, small talk, or general conversation.
        Return 'search' only for: questions about a specific topic, requests for information, or searching for data.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User query: {query}\n\nClassify the intent (greeting or search):")
        ])
        
        result = (prompt | llm_structured).invoke({"query": state["query"]})
        intent = result.intent if result else "search"
        
        return {"query_evaluator": intent, "messages": [HumanMessage(content=state["query"])]} # Temporarily storing the route decision in query_evaluator or we can make a new state key, let's just make the graph conditional edge read it.
    except Exception as e:
        print(f"Intent detector error: {e}")
        return {"query_evaluator": "search", "messages": [HumanMessage(content=state["query"])]}
