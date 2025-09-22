from sqlalchemy import Column, String, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
import uuid

from ..database import Base

class SubmissionStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Submission(Base):
    __tablename__ = "submissions"

    submission_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"))
    file_path = Column(String, nullable=False)   # blob storage path
    moderation_status = Column(String, default=SubmissionStatus.PENDING)
    appeal_status = Column(String, default="none")
