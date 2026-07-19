from datetime import datetime
from sqlmodel import Field, SQLModel, Column, JSON


class RagDocument(SQLModel, table=True):
    __tablename__ = "rag_documents"

    id: int | None = Field(default=None, primary_key=True)
    url: str = Field(nullable=False, index=True)
    title: str | None = Field(default=None)
    char_count: int = Field(default=0)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class RagChunk(SQLModel, table=True):
    __tablename__ = "rag_chunks"

    id: int | None = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="rag_documents.id", nullable=False, index=True)
    chunk_index: int = Field(nullable=False)
    content: str = Field(nullable=False)
    embedding: list[float] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
