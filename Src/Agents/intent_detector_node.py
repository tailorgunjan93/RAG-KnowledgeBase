from langchain_groq import ChatGroq
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from .Agent_state import state

# Using a faster/cheaper model for intent detection
class IntentClassification(BaseModel):
    intent: str = Field(description="Exactly 'greeting' or 'search', indicating if the user query is just saying hello/greeting, or actually searching/asking a question.")

def intent_detector(state: state):
    query = state["query"].strip().lower()
    
    # Heuristic check for common greetings (instant)
    common_greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you", "who are you"}
    if any(greet in query for greet in common_greetings) and len(query) < 20: # Limit length to ensure it's a simple greeting
        print("Heuristic greeting check triggered (Fast Path)!")
        return {"query_evaluator": "greeting"}

    try:
        # Prefer Groq for fast classification, fallback to Ollama
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model="llama3.2")
        else:
            model_name = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
            llm = ChatGroq(model=model_name, api_key=api_key)
            
        llm_structured = llm.with_structured_output(IntentClassification)
        
        system = """You are an intent classifier for a knowledge base agent.
        Determine if the user's message is a simple greeting/conversational pleasantry (like 'hello', 'hi', 'how are you', 'good morning') or an actual question/search query.
        Return 'greeting' if it is a simple greeting.
        Return 'search' if it contains a question, request, or keyword to look up."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User query: {query}\n\nClassify the intent (greeting or search):")
        ])
        
        result = (prompt | llm_structured).invoke({"query": query})
        intent = result.intent if result else "search"
        
        return {"query_evaluator": intent} # Temporarily storing the route decision in query_evaluator or we can make a new state key, let's just make the graph conditional edge read it.
    except Exception as e:
        print(f"Intent detector error: {e}")
        return {"query_evaluator": "search"}
