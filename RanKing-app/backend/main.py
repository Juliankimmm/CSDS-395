from fastapi import FastAPI
from routers import account, submission, voting, results, appeals

app = FastAPI(title="RanKing API")

# Register routers
app.include_router(account.router, prefix="/account", tags=["Account"])
app.include_router(submission.router, prefix="/submission", tags=["Submission"])
app.include_router(voting.router, prefix="/vote", tags=["Voting"])
app.include_router(results.router, prefix="/results", tags=["Results"])
app.include_router(appeals.router, prefix="/appeals", tags=["Appeals"])

@app.get("/")
def root():
    return {"message": "RanKing API is running"}
