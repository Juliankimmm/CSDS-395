from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class ContestCreate(BaseModel):
    name: str
    description: Optional[str] = None
    submission_start_date: datetime
    submission_end_date: datetime
    voting_end_date: datetime


class ContestResponse(BaseModel):
    contest_id: int
    name: str
    description: Optional[str] = None
    submission_start_date: datetime
    submission_end_date: datetime
    voting_end_date: datetime

    class Config:
        from_attributes = True