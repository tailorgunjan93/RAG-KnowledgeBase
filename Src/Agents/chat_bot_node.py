from .Agent_state import state
import os

def chat_bot(state: state):
    documents = state.get("document", [])
    query = state["query"]
    intent = state.get("query_evaluator", "search")
    
    docs = "\n".join([doc.page_content for doc, score in documents]) if documents else ""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.2")
    else:
        from langchain_groq import ChatGroq
        model_name = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
        llm = ChatGroq(model=model_name, api_key=api_key)
        
    if intent == "greeting":
        prompt = f"The user says: {state['query']}\nRespond warmly and briefly as an AI assistant. Do not mention that you don't have context, and keep it friendly!"
    else:
        prompt = f"Context: {docs}\n\nQuestion: {query}\nAnswer:"
        
    response = llm.invoke(prompt)
    return {"response": response.content}
