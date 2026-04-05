from fastapi import APIRouter,HTTPException
from Src.Evaluaters.QueryGrader import QueryGrader
from VectoreStore.faiss_search import search_faiss_index_with_score

GradeRouter = APIRouter()


@GradeRouter.get("/Grade")
async def GradeChecker(query: str):
    try:
        docs = search_faiss_index_with_score(query)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Vector search failed: {e!s}",
        ) from e

    out = []
    for doc, score in docs:
        try:
            grade = QueryGrader(query, doc.page_content)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Grader (Ollama / LLM) failed: {e!s}. "
                "Is Ollama running and is model `llama3.2` available?",
            ) from e
        out.append(
            {
                "document": doc.page_content,
                "score": float(score),
                "grade": grade,
            }
        )
    return out
