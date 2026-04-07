"""
ChatService — Thin wrapper around the LangGraph agent pipeline.

SRP: This service ONLY invokes the graph and extracts the response.
It does not build prompts, manage memory, or touch LLM config.

Usage:
    service = ChatService()
    answer = service.chat(query="hello", thread_id="user_123")
"""


class ChatService:
    """
    Thin interface to the LangGraph compiled agent (`app`).

    Decouples the HTTP router from the LangGraph internals.
    If the graph is restructured internally, only this file changes
    (not the router or test files).
    """

    def invoke(self, query: str, thread_id: str = "default") -> str:
        """
        Invoke the LangGraph agent and return the final text response.

        Args:
            query: The user's input query.
            thread_id: Conversation thread identifier for memory continuity.

        Returns:
            The agent's final response string.
        """
        # Import here to avoid circular imports at module load time
        from Src.Agents.Graph_builder import app

        config = {"configurable": {"thread_id": thread_id}}
        final_state = app.invoke({"query": query}, config=config)
        return final_state.get("response", "")
