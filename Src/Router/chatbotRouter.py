from Src.Agents.Graph_builder import app
from fastapi import APIRouter

chatbot_router = APIRouter()

@chatbot_router.get("/chat")
async def chat_bot(query: str, thread_id: str = "default_user_1"):
    # LangGraph expects the full state dict
    # the checkpointer requires a config with thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    final_state = app.invoke(
        {
            "query": query,
        },
        config=config
    )
    # Nodes store the model answer in state["response"] as a string.
    return {"answer": final_state.get("response", "")}