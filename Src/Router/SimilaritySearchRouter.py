from fastapi import APIRouter
from VectoreStore.faiss_search import search_dynamic_faiss_index_with_score
similarity_search_router = APIRouter()

@similarity_search_router.get("/similarity-search")
async def similarity_search_router_request(query: str):
    docs = search_dynamic_faiss_index_with_score(query)
    return [
            {
                "text": doc.page_content,
                "score": float(score),
                "file_id": doc.metadata.get("fileid"),
                "filename": doc.metadata.get("filename"),
            }
            for doc, score in docs
        ]
