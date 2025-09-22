from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # TODO: validate size, type, send to moderation
    return {"filename": file.filename, "status": "pending_moderation"}
