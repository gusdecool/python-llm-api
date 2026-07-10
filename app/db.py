from datetime import datetime
from sqlmodel import Field, SQLModel, create_engine, Session, select
from app.config import DATABASE_URL
from app.log import logger
from app.models import LLMJob, LLMMemory


# Create the SQLAlchemy engine for SQLModel
# We disable same-thread check for SQLite to allow multiple async/multithreaded requests
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL and DATABASE_URL.startswith("sqlite") else {},
    echo=False  # Set to True to print generated SQL to console
)



def init_db() -> None:
    """
    Initializes the database, creating all tables and seeding data if empty.
    This is safe function, can be called multiple times.
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