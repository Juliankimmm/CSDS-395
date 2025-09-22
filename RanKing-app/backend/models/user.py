from sqlalchemy import Column, String, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ..database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    bio = Column(String, default="")
    profile_pic = Column(String, nullable=True)   # URL to blob storage
    is_active = Column(Boolean, default=True)
    is_private = Column(Boolean, default=False)
