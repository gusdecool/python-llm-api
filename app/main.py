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


# def get_db():
#     """
#     FastAPI dependency that yields a database connection.
#     """
#     conn = sqlite3.connect(DATABASE_URL)
#     conn.row_factory = sqlite3.Row
#     try:
#         yield conn
#     finally:
#         conn.close()


# Pydantic models for request validation and response schema
# class ItemBase(BaseModel):
#     name: str = Field(..., min_length=1, max_length=100, description="The name of the item")
#     description: Optional[str] = Field(None, max_length=250, description="The description of the item")
#     price: float = Field(..., gt=0, description="The price of the item, must be greater than zero")
#     tax: Optional[float] = Field(None, ge=0, description="The tax applied to the item")


# class ItemCreate(ItemBase):
#     pass


# class ItemResponse(ItemBase):
#     id: int = Field(..., description="The unique identifier of the item")




# @app.get("/items", response_model=List[ItemResponse], tags=["Items"])
# def get_items(limit: int = 10, conn: sqlite3.Connection = Depends(get_db)) -> List[dict]:
#     """
#     Retrieve a list of items.
    
#     Query parameter:
#     - **limit**: Maximum number of items to return (default is 10)
#     """
#     log.info("Fetching items up to limit: %d", limit)
#     cursor = conn.cursor()
#     cursor.execute("SELECT id, name, description, price, tax FROM items LIMIT ?", (limit,))
#     rows = cursor.fetchall()
#     return [dict(row) for row in rows]


# @app.post(
#     "/items",
#     response_model=ItemResponse,
#     status_code=status.HTTP_201_CREATED,
#     tags=["Items"]
# )
# def create_item(item: ItemCreate, conn: sqlite3.Connection = Depends(get_db)) -> dict:
#     """
#     Create a new item.
    
#     Accepts validated request body, assigns a new ID, saves in database,
#     and returns the created item.
#     """
#     log.info("Creating a new item: %s", item.name)
#     cursor = conn.cursor()
#     cursor.execute(
#         "INSERT INTO items (name, description, price, tax) VALUES (?, ?, ?, ?)",
#         (item.name, item.description, item.price, item.tax)
#     )
#     conn.commit()
#     new_id = cursor.lastrowid
    
#     cursor.execute("SELECT id, name, description, price, tax FROM items WHERE id = ?", (new_id,))
#     row = cursor.fetchone()
#     return dict(row)



