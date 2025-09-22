from sqlalchemy import Column, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ..database import Base

class Appeal(Base):
    __tablename__ = "appeals"

    appeal_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.submission_id"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"))
    status = Column(String, default="pending")  # pending, approved, denied
    decision_reason = Column(String, nullable=True)
