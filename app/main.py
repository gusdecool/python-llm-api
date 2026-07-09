import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fastapi-demo")

app = FastAPI(
    title="FastAPI Simple Demo API",
    description="A simple HTTP GET and POST REST API example to demonstrate request and response handling.",
    version="1.0.0"
)

# In-memory database representation
items_db: Dict[int, dict] = {
    1: {"id": 1, "name": "Item One", "description": "The first item", "price": 49.99, "tax": 4.0},
    2: {"id": 2, "name": "Item Two", "description": "The second item", "price": 99.50, "tax": 8.0},
}


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
def get_items(limit: int = 10) -> List[dict]:
    """
    Retrieve a list of items.
    
    Query parameter:
    - **limit**: Maximum number of items to return (default is 10)
    """
    logger.info("Fetching items up to limit: %d", limit)
    return list(items_db.values())[:limit]


@app.post(
    "/items",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Items"]
)
def create_item(item: ItemCreate) -> dict:
    """
    Create a new item.
    
    Accepts validated request body, assigns a new ID, saves in-memory,
    and returns the created item.
    """
    logger.info("Creating a new item: %s", item.name)
    
    # Simple ID generation
    new_id = max(items_db.keys()) + 1 if items_db else 1
    
    new_item = {
        "id": new_id,
        "name": item.name,
        "description": item.description,
        "price": item.price,
        "tax": item.tax
    }
    
    items_db[new_id] = new_item
    return new_item
