from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import shutil
import uuid
import os

from ..database import get_db
from ..models.submission import Submission
from ..services.moderation_service import ModerationService

router = APIRouter()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_image(user_id: str, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    # 1. Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file format")

    # 2. Validate size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # 3. Save to disk (later → Azure Blob Storage)
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(contents)

    # 4. Run moderation (mock service for now)
    moderation_result = ModerationService.analyze_image(contents)

    # 5. Store submission in DB
    submission = Submission(user_id=user_id, file_path=filepath, moderation_status=moderation_result)
    db.add(submission)
    await db.commit()
    await db.refresh(submission)

    if moderation_result != "safe":
        return {"status": "rejected", "reason": moderation_result}
    return {"status": "accepted", "submission_id": str(submission.submission_id)}
