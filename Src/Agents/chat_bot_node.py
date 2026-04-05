from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .Agent_state import AgentState
import os

def chat_bot(state: AgentState):
    documents = state.get("document", [])
    query = state["query"]
    intent = state.get("query_evaluator", "search")
    history = state.get("messages", [])
    
    docs = "\n".join([doc.page_content for doc, score in documents]) if documents else "No relevant documents found."
    
    from Src.Utils.llm_utils import get_llm
    llm = get_llm(performance="high")
        
    if intent == "greeting":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Respond warmly and briefly as an AI assistant. Stay conversational and friendly!"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant. Use the following context to answer the user's question.\n\nContext:\n{docs}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
    chain = prompt | llm
    response = chain.invoke({"history": history, "query": query})
    
    return {
        "response": response.content,
        "messages": [AIMessage(content=response.content)]
    }
