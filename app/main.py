import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fastapi-demo")

DATABASE_URL = "items.db"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that handles startup and shutdown tasks.
    """
    init_db()
    yield


app = FastAPI(
    title="FastAPI Simple Demo API",
    description="A simple HTTP GET and POST REST API example to demonstrate request and response handling.",
    version="1.0.0",
    lifespan=lifespan
)


def init_db(db_path: str = DATABASE_URL) -> None:
    """
    Initializes the SQLite database. Creates the items table if it doesn't exist
    and seeds it with initial items if empty.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                price REAL NOT NULL,
                tax REAL
            )
        """)
        
        # Check if table is empty to seed it
        cursor.execute("SELECT COUNT(*) FROM items")
        if cursor.fetchone()[0] == 0:
            cursor.executemany("""
                INSERT INTO items (name, description, price, tax)
                VALUES (?, ?, ?, ?)
            """, [
                ("Item One", "The first item", 49.99, 4.0),
                ("Item Two", "The second item", 99.50, 8.0)
            ])
            conn.commit()
            logger.info("Database initialized and seeded.")
        else:
            logger.info("Database already initialized.")
    finally:
        conn.close()


def get_db():
    """
    FastAPI dependency that yields a database connection.
    """
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# Pydantic models for request validation and response schema
class ItemBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="The name of the item")
    description: Optional[str] = Field(None, max_length=250, description="The description of the item")
    price: float = Field(..., gt=0, description="The price of the item, must be greater than zero")
    tax: Optional[float] = Field(None, ge=0, description="The tax applied to the item")


class ItemCreate(ItemBase):
    pass


class ItemResponse(ItemBase):
    id: int = Field(..., description="The unique identifier of the item")


@app.get("/", tags=["Root"])
def read_root() -> Dict[str, str]:
    """
    Root endpoint offering a simple welcome message.
    """
    return {"message": "Welcome to the FastAPI Demo API. Visit /docs for interactive documentation."}


@app.get("/items", response_model=List[ItemResponse], tags=["Items"])
def get_items(limit: int = 10, conn: sqlite3.Connection = Depends(get_db)) -> List[dict]:
    """
    Retrieve a list of items.
    
    Query parameter:
    - **limit**: Maximum number of items to return (default is 10)
    """
    logger.info("Fetching items up to limit: %d", limit)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, description, price, tax FROM items LIMIT ?", (limit,))
    rows = cursor.fetchall()
    return [dict(row) for row in rows]


@app.post(
    "/items",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Items"]
)
def create_item(item: ItemCreate, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    """
    Create a new item.
    
    Accepts validated request body, assigns a new ID, saves in database,
    and returns the created item.
    """
    logger.info("Creating a new item: %s", item.name)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO items (name, description, price, tax) VALUES (?, ?, ?, ?)",
        (item.name, item.description, item.price, item.tax)
    )
    conn.commit()
    new_id = cursor.lastrowid
    
    cursor.execute("SELECT id, name, description, price, tax FROM items WHERE id = ?", (new_id,))
    row = cursor.fetchone()
    return dict(row)
