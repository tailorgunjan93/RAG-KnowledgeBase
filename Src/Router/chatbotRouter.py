"""
chatbotRouter — Thin HTTP adapter for conversational queries.

Delegates all business logic to ChatService via the DI container.
"""
from fastapi import APIRouter, HTTPException

from Src.container import chat_service

chatbot_router = APIRouter(tags=["Chat"])


@chatbot_router.get("/chat")
async def chat_bot(query: str, thread_id: str = "default_user_1"):
    """
    Submit a query to the RAG agent and get a response.

    The thread_id enables persistent conversation memory across requests.
    Use a consistent thread_id per user/session to maintain context.
    """
    try:
        answer = chat_service.invoke(query=query, thread_id=thread_id)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}")