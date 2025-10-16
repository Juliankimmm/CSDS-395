from pydantic import BaseModel
from datetime import datetime


class VoteResponse(BaseModel):
    vote_id: int
    user_id: int
    submission_id: int
    voted_at: datetime

    class Config:
        from_attributes = True