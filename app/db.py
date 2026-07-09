from datetime import datetime
from sqlmodel import Field, SQLModel, create_engine, Session, select
from app.config import DATABASE_URL
from app.log import logger

# Create the SQLAlchemy engine for SQLModel
# We disable same-thread check for SQLite to allow multiple async/multithreaded requests
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=True  # Prints generated SQL to console (good for development)
)


class LLMJob(SQLModel, table=True):
    __tablename__ = "llm_jobs"

    id: int | None = Field(default=None, primary_key=True)
    prompt: str = Field(nullable=False)
    response: str | None = Field(default=None)
    status: str = Field(default="done", max_length=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)


def init_db() -> None:
    """
    Initializes the database, creating all tables and seeding data if empty.
    """
    # Create tables
    SQLModel.metadata.create_all(engine)

    # Seed initial data if table is empty
    with Session(engine) as session:
        statement = select(LLMJob)
        if not session.exec(statement).first():
            job1 = LLMJob(prompt="what is E=mc2", status="queue")
            job2 = LLMJob(prompt="who is the genius", response="It's you Budi", status="done")
            session.add(job1)
            session.add(job2)
            session.commit()
            logger.info("Database initialized and seeded with SQLModel.")
        else:
            logger.info("Database already initialized.")


def get_session():
    """
    Dependency to yield database sessions for requests.
    """
    with Session(engine) as session:
        yield session