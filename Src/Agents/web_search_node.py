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

    llm = ChatOllama(model="llama3.2")
    prompt = f"Context (Web Search Results): {docs}\n\nQuestion: {query}\nProvide a clear and concise answer based on the web search results above:"
    response = llm.invoke(prompt)
    return {"response": response.content}
