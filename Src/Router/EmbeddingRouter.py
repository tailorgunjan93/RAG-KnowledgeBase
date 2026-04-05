from fastapi import APIRouter
from pathlib import Path
from Src.Embeddings.embeddings import embed_file
embedding_router = APIRouter()

@embedding_router.get("/embed/{file_name}")
async def embed_file_router(file_name: str):
    try:
        file_path = Path(__file__).parent.parent / "Uploads" / f"{file_name}"
        with open(file_path, "rb") as f:
            content = f.read()         
        faiss_index = await embed_file(file_path)
        return {"message":"fileExists","faiss_index":faiss_index.index.ntotal}
    except FileNotFoundError:
        return {"message":"fileNotExists","faiss_index":None}