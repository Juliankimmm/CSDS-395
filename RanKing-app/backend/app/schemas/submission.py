from pydantic import BaseModel
from datetime import datetime


class SubmissionResponse(BaseModel):
    sub_id: int
    user_id: int
    contest_id: int
    image_path: str
    submitted_at: datetime
    vote_count: int = 0

    class Config:
        from_attributes = True