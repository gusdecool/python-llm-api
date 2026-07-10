# FastAPI Simple REST API Demo

A clean, standard Python FastAPI project showcasing basic HTTP GET and POST request and response handling, with automated validation via Pydantic.

## Features
1. Multi agentic agent, there is an Agentic AI agent that will chose which agent to use depend on the task.
2. Generate image agent
3. Weather agent
4. Memory agent (remember who you are or what you asked for). Good for saving memory or API call.

## Dependencies

The project defines the following packages in `requirements.txt`:

1. **`fastapi`**: The core web framework for building APIs with Python 3.8+ based on standard Python type hints. It provides high performance (on par with NodeJS and Go), rapid coding, and automatic interactive Swagger/ReDoc documentation.
2. **`uvicorn[standard]`**: An ASGI (Asynchronous Server Gateway Interface) web server implementation for Python. FastAPI is built on ASGI standard, and `uvicorn` acts as the server to run the FastAPI application. The `[standard]` extra installs high-performance loop dependencies like `uvloop` and `httptools`.
3. **`pydantic`**: Data validation and settings management using Python type annotations. FastAPI uses Pydantic to parse and validate request JSON payloads, and serialize response objects.
4. **`pytest`**: A robust testing framework for writing clean, readable, and scalable unit tests.
5. **`httpx`**: A next-generation HTTP client for Python. It is used in unit tests alongside FastAPI's `TestClient` to make asynchronous/synchronous HTTP requests to the application.

---

## Getting Started

### 1. Prerequisites
- Python 3.8 or higher.

### 2. Setup Virtual Environment
Run the following commands inside the project:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# create an empty sqlite db
touch main.db
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Running the API Server
Start the development server with automatic reload on code changes:
```bash
uvicorn app.main:app --reload
```
By default, the server runs on `http://127.0.0.1:8000`.

---

## Interactive API Documentation

FastAPI automatically generates interactive API documentation. Once the server is running, visit:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## Running Automated Tests

Run the test suite using `pytest` with `PYTHONPATH` set to the project root:
```bash
PYTHONPATH=. pytest
```
