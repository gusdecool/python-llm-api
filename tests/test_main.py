from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Welcome" in response.json()["message"]


def test_get_items():
    response = client.get("/items")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    # Verify properties of elements
    assert data[0]["name"] == "Item One"
    assert "id" in data[0]


def test_create_item_success():
    payload = {
        "name": "New Test Item",
        "description": "A testing item",
        "price": 19.99,
        "tax": 1.50
    }
    response = client.post("/items", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == payload["name"]
    assert "id" in data
    assert data["price"] == payload["price"]


def test_create_item_invalid_price():
    payload = {
        "name": "Invalid Item",
        "price": -5.00  # Invalid since price must be > 0
    }
    response = client.post("/items", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
