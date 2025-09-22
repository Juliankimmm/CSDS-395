from fastapi import APIRouter

router = APIRouter()

@router.get("/leaderboard")
def leaderboard(round_id: str):
    # TODO: query DB for top ranked outfits
    return {"leaderboard": []}
