from langchain_ollama import ChatOllama

from .Agent_state import state

def chat_bot(state: state):
    documents = state["document"]
    query = state["query"]
    docs = "\n".join([doc.page_content for doc, score in documents])
    llm = ChatOllama(model="llama3.2")
    prompt = f"Context: {docs}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)
    return {"response": response.content}
