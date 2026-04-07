"""
GraderRouter — Thin HTTP adapter for document relevance grading.

Delegates all business logic to SearchService and QueryGrader via the DI container.
"""
from fastapi import APIRouter, HTTPException

from Src.container import search_service
from Src.Evaluaters.QueryGrader import QueryGrader

GradeRouter = APIRouter(tags=["Grader"])


@GradeRouter.get("/Grade")
async def GradeChecker(query: str):
    """
    Retrieve documents and grade their relevance to the given query.

    Useful for debugging retrieval quality — shows FAISS scores and LLM grading.
    """
    try:
        docs = search_service.retrieve(query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Vector search failed: {exc}")

    out = []
    for doc, score in docs:
        try:
            grade = QueryGrader(query, doc.page_content)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Grader failed: {exc}",
            )
        out.append({
            "document": doc.page_content,
            "score": float(score),
            "grade": grade,
        })

    return out
