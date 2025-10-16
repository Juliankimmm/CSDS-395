from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from ..database import get_db
from ..models.models import Contest
from ..schemas.contest import ContestCreate, ContestResponse

router = APIRouter(prefix="/contests", tags=["Contests"])


@router.get("", response_model=List[ContestResponse])
def get_contests(
    status: Optional[str] = None, db: Session = Depends(get_db)
):
    """Get all contests, optionally filtered by status"""
    query = db.query(Contest)
    now = datetime.utcnow()

    if status == "submission":
        query = query.filter(
            Contest.submission_start_date <= now,
            Contest.submission_end_date >= now,
        )
    elif status == "voting":
        query = query.filter(
            Contest.submission_end_date < now, Contest.voting_end_date >= now
        )
    elif status == "finished":
        query = query.filter(Contest.voting_end_date < now)

    return query.all()


@router.get("/{contest_id}", response_model=ContestResponse)
def get_contest(contest_id: int, db: Session = Depends(get_db)):
    """Get detailed information for a single contest"""
    contest = (
        db.query(Contest).filter(Contest.contest_id == contest_id).first()
    )
    if not contest:
        raise HTTPException(status_code=404, detail="Contest not found")
    return contest


@router.post("", response_model=ContestResponse, status_code=201)
def create_contest(contest: ContestCreate, db: Session = Depends(get_db)):
    """Create a new contest (admin functionality)"""
    db_contest = Contest(**contest.model_dump())
    db.add(db_contest)
    db.commit()
    db.refresh(db_contest)
    return db_contest