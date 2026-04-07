"""
UploadRouter — Thin HTTP adapter for file uploads.

Saves files to the configured uploads directory.
No business logic here — this is purely an I/O endpoint.
"""
from fastapi import APIRouter, HTTPException, UploadFile

from Src.Config.settings import settings

upload_router = APIRouter(tags=["Upload"])


@upload_router.post("/upload")
async def upload_file(file: UploadFile):
    """
    Upload a file to the server for later embedding.

    After uploading, call GET /embed/{filename} to process the document.
    """
    try:
        settings.uploads_path.mkdir(parents=True, exist_ok=True)
        file_path = settings.uploads_path / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        return {"message": "File uploaded successfully", "file": file.filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")
