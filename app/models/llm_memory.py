from datetime import datetime
from sqlmodel import Field, SQLModel


class LLMMemory(SQLModel, table=True):
    __tablename__ = "llm_memories"

    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(nullable=False, index=True)
    memory_type: str = Field(nullable=False, index=True)  # 'profile', 'weather', 'image', 'direct_answer'
    query_key: str = Field(nullable=False, index=True)    # Normalized prompt, key, or city name
    response_val: str = Field(nullable=False)            # Response text or JSON payload
    created_at: datetime = Field(default_factory=datetime.utcnow)
