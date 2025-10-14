from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    status,
)
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import datetime
from ..database import get_db
from ..models.models import User, Contest, Submission, Vote
from ..schemas.submission import SubmissionResponse
from ..dependencies import get_current_user
from ..utils.file_handler import save_upload_file

router = APIRouter(tags=["Submissions"])


@router.post(
    "/contests/{contest_id}/submissions",
    response_model=SubmissionResponse,
    status_code=201,
)
def create_submission(
    contest_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new submission for a contest"""
    # Check if contest exists
    contest = (
        db.query(Contest).filter(Contest.contest_id == contest_id).first()
    )
    if not contest:
        raise HTTPException(status_code=404, detail="Contest not found")

    # Check if submission period is active
    now = datetime.utcnow()
    if not (
        contest.submission_start_date <= now <= contest.submission_end_date
    ):
        raise HTTPException(
            status_code=400,
            detail="Contest is not accepting submissions at this time",
        )

    # Check if user already submitted
    existing_submission = (
        db.query(Submission)
        .filter(
            Submission.user_id == current_user.user_id,
            Submission.contest_id == contest_id,
        )
        .first()
    )
    if existing_submission:
        raise HTTPException(
            status_code=400,
            detail="You have already submitted to this contest",
        )

    # Save file
    try:
        file_path = save_upload_file(file, contest_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {str(e)}"
        )

    # Create submission
    submission = Submission(
        user_id=current_user.user_id,
        contest_id=contest_id,
        image_path=file_path,
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    return submission


@router.get(
    "/contests/{contest_id}/submissions",
    response_model=List[SubmissionResponse],
)
def get_contest_submissions(
    contest_id: int, db: Session = Depends(get_db)
):
    """Get all submissions for a contest"""
    contest = (
        db.query(Contest).filter(Contest.contest_id == contest_id).first()
    )
    if not contest:
        raise HTTPException(status_code=404, detail="Contest not found")

    # Get submissions with vote counts
    submissions = (
        db.query(
            Submission, func.count(Vote.vote_id).label("vote_count")
        )
        .outerjoin(Vote, Vote.submission_id == Submission.sub_id)
        .filter(Submission.contest_id == contest_id)
        .group_by(Submission.sub_id)
        .all()
    )

    return [
        SubmissionResponse(
            sub_id=sub.sub_id,
            user_id=sub.user_id,
            contest_id=sub.contest_id,
            image_path=sub.image_path,
            submitted_at=sub.submitted_at,
            vote_count=count,
        )
        for sub, count in submissions
    ]