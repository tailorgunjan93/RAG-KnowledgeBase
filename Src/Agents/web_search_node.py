from langchain_ollama import ChatOllama

from Src.Tools.internet_search_tool import create_search_results

from .Agent_state import state


def web_search(state: state):
    query = state["query"]
    search_results = create_search_results(query)

    docs = (
        "\n".join(search_results)
        if isinstance(search_results, list)
        else str(search_results)
    )

    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.2")
    else:
        from langchain_groq import ChatGroq
        model_name = os.getenv("GROQ_MODEL_NAME", "llama3-8b-8192")
        llm = ChatGroq(model=model_name, api_key=api_key)
    prompt = f"Context (Web Search Results): {docs}\n\nQuestion: {query}\nProvide a clear and concise answer based on the web search results above:"
    response = llm.invoke(prompt)
    return {"response": response.content}
