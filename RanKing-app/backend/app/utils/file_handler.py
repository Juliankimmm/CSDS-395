import os
import uuid
from fastapi import UploadFile, HTTPException
from ..config import settings


def save_upload_file(upload_file: UploadFile, contest_id: int) -> str:
    """Save uploaded file and return the file path"""
    # Validate file size
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB

    # Create upload directory if it doesn't exist
    contest_dir = os.path.join(
        settings.upload_dir, f"contest_{contest_id}"
    )
    os.makedirs(contest_dir, exist_ok=True)

    # Generate unique filename
    file_extension = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(contest_dir, unique_filename)

    # Save file
    with open(file_path, "wb") as f:
        while chunk := upload_file.file.read(chunk_size):
            file_size += len(chunk)
            if file_size > settings.max_file_size:
                os.remove(file_path)
                raise HTTPException(
                    status_code=413, detail="File too large"
                )
            f.write(chunk)

    return file_path


def delete_file(file_path: str):
    """Delete a file from filesystem"""
    if os.path.exists(file_path):
        os.remove(file_path)