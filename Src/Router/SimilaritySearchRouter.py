from fastapi import APIRouter
from VectoreStore.faiss_search import search_faiss_index,search_faiss_index_with_score,search_faiss_index_with_score_and_metadata

similarity_search_router = APIRouter()

@similarity_search_router.get("/similarity-search")
async def similarity_search_router_request(query: str):
    docs = search_faiss_index_with_score(query)
    return [
            {
                "text": doc.page_content,
                "score": float(score),
                "file_id": doc.metadata.get("fileid"),
                "filename": doc.metadata.get("filename"),
            }
            for doc, score in docs
        ]
