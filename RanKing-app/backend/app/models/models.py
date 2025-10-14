from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from ..database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    submissions = relationship("Submission", back_populates="user")
    votes = relationship("Vote", back_populates="user")


class Contest(Base):
    __tablename__ = "contests"

    contest_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    submission_start_date = Column(DateTime, nullable=False)
    submission_end_date = Column(DateTime, nullable=False)
    voting_end_date = Column(DateTime, nullable=False)

    submissions = relationship("Submission", back_populates="contest")


class Submission(Base):
    __tablename__ = "submissions"

    sub_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    contest_id = Column(
        Integer, ForeignKey("contests.contest_id"), nullable=False
    )
    image_path = Column(String(255), nullable=False)
    submitted_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="submissions")
    contest = relationship("Contest", back_populates="submissions")
    votes = relationship("Vote", back_populates="submission")


class Vote(Base):
    __tablename__ = "votes"

    vote_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    submission_id = Column(
        Integer, ForeignKey("submissions.sub_id"), nullable=False
    )
    voted_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="votes")
    submission = relationship("Submission", back_populates="votes")

    __table_args__ = (
        UniqueConstraint(
            "user_id", "submission_id", name="unique_user_submission"
        ),
    )