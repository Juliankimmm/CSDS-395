from fastapi import APIRouter

router = APIRouter()

@router.post("/submit")
def submit_appeal(submission_id: str, reason: str):
    # TODO: log appeal in DB
    return {"message": "Appeal submitted"}
