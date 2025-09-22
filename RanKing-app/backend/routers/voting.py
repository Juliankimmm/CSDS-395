from fastapi import APIRouter

router = APIRouter()

@router.post("/cast")
def cast_vote(user_id: str, submission_id: str):
    # TODO: prevent self-vote, log vote
    return {"message": "Vote cast successfully"}
