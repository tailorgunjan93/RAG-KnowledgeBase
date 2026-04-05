from fastapi import APIRouter,Query,UploadFile
from pathlib import Path

upload_router = APIRouter()

@upload_router.post("/upload")
async def upload_file(file: UploadFile):
    file_path = Path(__file__).parent.parent / "Uploads" / f"{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": "File uploaded successfully"}
    
