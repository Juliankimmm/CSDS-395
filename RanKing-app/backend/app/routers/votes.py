from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from ..database import get_db
from ..models.models import User, Submission, Vote, Contest
from ..schemas.vote import VoteResponse
from ..dependencies import get_current_user

router = APIRouter(tags=["Votes"])


@router.post(
    "/submissions/{submission_id}/vote",
    response_model=VoteResponse,
    status_code=201,
)
def vote_for_submission(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cast a vote for a submission"""
    # Check if submission exists
    submission = (
        db.query(Submission)
        .filter(Submission.sub_id == submission_id)
        .first()
    )
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Check if voting period is active
    contest = (
        db.query(Contest)
        .filter(Contest.contest_id == submission.contest_id)
        .first()
    )
    now = datetime.utcnow()
    if not (contest.submission_end_date < now <= contest.voting_end_date):
        raise HTTPException(
            status_code=400,
            detail="Voting is not active for this contest",
        )

    # Check if user is trying to vote for their own submission
    if submission.user_id == current_user.user_id:
        raise HTTPException(
            status_code=400, detail="Cannot vote for your own submission"
        )

    # Check if user already voted for this submission
    existing_vote = (
        db.query(Vote)
        .filter(
            Vote.user_id == current_user.user_id,
            Vote.submission_id == submission_id,
        )
        .first()
    )
    if existing_vote:
        raise HTTPException(
            status_code=400,
            detail="You have already voted for this submission",
        )

    # Create vote
    vote = Vote(user_id=current_user.user_id, submission_id=submission_id)
    db.add(vote)
    db.commit()
    db.refresh(vote)

    return vote