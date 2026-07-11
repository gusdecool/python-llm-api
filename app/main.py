# import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from app.log import logger
from app.db import init_db
from app.routes.llm_job import router as llm_job_router

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("fastapi-demo")

# DATABASE_URL = "items.db"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that handles startup and shutdown tasks.
    """
    # put anything that should run during init here
    yield


app = FastAPI(
    title="LLM FastApi",
    description="A super simple FastApi to send prompt to llm.",
    version="1.0.0",
    lifespan=lifespan
)


app.include_router(llm_job_router)


@app.get("/", tags=["Root"])
def read_root() -> Dict[str, str]:
    """
    Root endpoint offering a simple welcome message.
    """
    return {"message": "Welcome to the LLM API, I'm here to help'."}


@app.post("/db-init", tags=["Database"])
def db_init() -> dict:
    """
    Initialize the database.
    """
    logger.info("Initializing database...")
    init_db()
    # with get_db_conn() as conn:
    #     init_db(conn)

    return {"message": "Database initialized successfully."}

