from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta
from ..database import get_db
from ..models.models import User
from ..schemas.user import UserCreate, UserLogin, UserResponse, Token
from ..utils.security import (
    get_password_hash,
    verify_password,
    create_access_token,
)
from ..config import settings

router = APIRouter(prefix="", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=201)
def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user exists
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(
            status_code=400, detail="Email already registered"
        )
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(
            status_code=400, detail="Username already taken"
        )

    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        password_hash=hashed_password,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


@router.post("/login", response_model=Token)
def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access token"""
    user = db.query(User).filter(User.email == user_credentials.email).first()

    if not user or not verify_password(
        user_credentials.password, user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    access_token_expires = timedelta(
        minutes=settings.access_token_expire_minutes
    )
    access_token = create_access_token(
        data={"sub": str(user.user_id)}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}