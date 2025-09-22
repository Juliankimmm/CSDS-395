from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class UserCreate(BaseModel):
    email: str
    password: str

@router.post("/register")
def register(user: UserCreate):
    # TODO: implement DB insert, bcrypt hash, OAuth2
    return {"message": f"User {user.email} registered successfully"}

@router.post("/login")
def login(user: UserCreate):
    # TODO: implement OAuth2 login
    return {"message": f"User {user.email} logged in"}
