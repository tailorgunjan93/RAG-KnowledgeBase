from Src.Tools.internet_search_tool import create_search_results
from .Agent_state import AgentState
from Src.Utils.llm_utils import get_llm

def web_search(state: AgentState):
    query = state["query"]
    print(f"--- Web Search started for: {query} ---")
    
    try:
        search_results = create_search_results(query)
        docs = (
            "\n".join(search_results)
            if isinstance(search_results, list)
            else str(search_results)
        )
    except Exception as e:
        print(f"Web search tool error: {e}")
        docs = f"Search failed: {e}"

    llm = get_llm(performance="standard")
    prompt = f"Context (Web Search Results): {docs}\n\nQuestion: {query}\nProvide a clear and concise answer based on the web search results above:"
    
    try:
        response = llm.invoke(prompt)
        return {"response": response.content}
    except Exception as e:
        print(f"LLM invoke error in web_search: {e}")
        return {"response": "I'm having trouble reasoning through the search results. Please try again."}
