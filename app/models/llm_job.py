from datetime import datetime
from sqlmodel import Field, SQLModel

class LLMJob(SQLModel, table=True):
    __tablename__ = "llm_jobs"

    id: int | None = Field(default=None, primary_key=True)
    prompt: str = Field(nullable=False)
    response: str | None = Field(default=None)
    status: str = Field(default="done", max_length=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    responded_at: datetime | None = Field(default=None)