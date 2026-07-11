# Python LLM Multi Agents

A clean, python implementation to orchestrated multi Agentic AI with LangChain/LangGraph & Langfuse (observability).

## Features
1. Multi Agents, system will chose which AI agent suitable for the tasks.
2. Memory capability.
3. Model agnostic, easily switch between model that suit the task.
4. Multi platform: CLI, MCP Server, RESTful API.

## Available Agents
- `agent_chooser`: decide which agent specialized agent to use. If can answer directly (e.g: simple question, answer directly)
- `agent_memory`: agent that remember the previous question and use it as base answer. Not to be confused with full RAG capability.
- `weather_agent`: agent that can answer weather related question, powered by Open Weather API
- `image_agent`: agent that can generate image.

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

Create the `.env` file, see file [config.py](./app/config.py) for env used.

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Running the API Server (REST)
Start the development server with automatic reload on code changes:
```bash
uvicorn app.main:app --reload
```
By default, the server runs on `http://127.0.0.1:8000`.

Interactive API Documentation
FastAPI automatically generates interactive API documentation. Once the server is running, visit:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 5. Running as CLI (recommended)
```bash
python -m cli
```

### 6. As MCP server

#### Option 1: Standard Input/Output (stdio) - Best for local use
To run the server over stdio:
```bash
./venv/bin/python mcp_server.py
```

**Claude Desktop Configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "agent-orchestrator": {
      "command": "<directory>/python-llm-api/venv/bin/python",
      "args": [
        "<directory>/python-llm-api/mcp_server.py"
      ]
    }
  }
}
```

#### Option 2: Server-Sent Events (SSE) - Best for deployment / URL-based access
To start the server over HTTP/SSE (exposed via a URL on `http://localhost:8000/sse`):
```bash
python -m mcp_server sse
```

**Claude Desktop Configuration (`~/Library/Application Support/Claude/claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "agent-orchestrator": {
      "ServerUrl": "http://localhost:8000/sse"
    }
  }
}
```

**Cursor Configuration:**
Go to **Settings > Features > MCP**:
- Click **+ Add New MCP Server**
- Name: `agent-orchestrator`
- Type: `SSE`
- URL: `http://localhost:8000/sse`

---

## Running Automated Tests

Run the test suite using `pytest` with `PYTHONPATH` set to the project root:
```bash
PYTHONPATH=. pytest
```

## Dependencies

The project defines the following packages in `requirements.txt`:

1. **`fastapi`**: The core web framework for building APIs with Python 3.8+ based on standard Python type hints. It provides high performance (on par with NodeJS and Go), rapid coding, and automatic interactive Swagger/ReDoc documentation.
2. **`uvicorn[standard]`**: An ASGI (Asynchronous Server Gateway Interface) web server implementation for Python. FastAPI is built on ASGI standard, and `uvicorn` acts as the server to run the FastAPI application. The `[standard]` extra installs high-performance loop dependencies like `uvloop` and `httptools`.
3. **`pydantic`**: Data validation and settings management using Python type annotations. FastAPI uses Pydantic to parse and validate request JSON payloads, and serialize response objects.
4. **`pytest`**: A robust testing framework for writing clean, readable, and scalable unit tests.
5. **`httpx`**: A next-generation HTTP client for Python. It is used in unit tests alongside FastAPI's `TestClient` to make asynchronous/synchronous HTTP requests to the application.