"""
EmbeddingRouter — Thin HTTP adapter for file embedding.

Delegates all business logic to EmbeddingService via the DI container.
This router has zero knowledge of FAISS, Neo4j, or LLM configuration.
"""
from pathlib import Path

from fastapi import APIRouter, HTTPException

from Src.container import make_embedding_service
from Src.Config.settings import settings

embedding_router = APIRouter(tags=["Embedding"])


@embedding_router.get("/embed/{file_name}")
async def embed_file_router(file_name: str):
    """
    Trigger ingestion of an already-uploaded file.

    The file must exist in the Uploads directory (use /upload first).
    Returns the total number of vector chunks stored after ingestion.
    """
    file_path = settings.uploads_path / file_name

    try:
        service = make_embedding_service(index_name=Path(file_name).stem)
        count = await service.process_file(file_path)
        return {"message": "success", "file": file_name, "document_count": count}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found in uploads.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")