"""
SimilaritySearchRouter — Thin HTTP adapter for raw vector similarity search.

Delegates all business logic to SearchService via the DI container.
"""
from fastapi import APIRouter, HTTPException

from Src.container import search_service

similarity_search_router = APIRouter(tags=["Search"])


@similarity_search_router.get("/similarity-search")
async def similarity_search_router_request(query: str, k: int = 4):
    """
    Raw semantic similarity search against the vector store.

    Returns ranked document chunks and their similarity scores.
    Lower score = more similar (L2 distance).
    """
    try:
        docs = search_service.retrieve(query, k=k)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search failed: {exc}")

    return [
        {
            "text": doc.page_content,
            "score": float(score),
            "file_id": doc.metadata.get("fileid"),
            "filename": doc.metadata.get("filename"),
        }
        for doc, score in docs
    ]
