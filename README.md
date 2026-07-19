# Python LLM Multi Agents

A clean, python implementation to orchestrated multi Agentic AI with LangChain/LangGraph & Langfuse (observability).

## Features
1. Multi Agents, the system will choose which AI agent is suitable for the tasks.
2. Memory capability.
3. Model agnostic, easily switch between models that suits the task.
4. Multi platform: CLI, MCP Server, RESTful API.
5. On-demand RAG knowledge base: add any URL and query its content afterward.

## Available Agents
- `agent_chooser`: decide which agent specialized agent to use. If can answer directly (e.g: simple question, answer directly)
- `agent_memory`: agent that remember the previous question and use it as base answer. Not to be confused with full RAG capability.
- `weather_agent`: agent that can answer weather related question, powered by Open Weather API
- `image_agent`: agent that can generate image.
- `rag_ingest_agent`: scrapes a given URL, chunks and embeds its content, and stores it in the knowledge base (SQLite). Skips re-scraping if the URL was already added.
- `rag_query_agent`: answers questions by retrieving the most relevant chunks from the knowledge base (via embedding similarity) and synthesizing an answer from them. Replies that it doesn't know if nothing relevant has been added yet.

### RAG Knowledge Base Example
```
You: Add https://www.kayak.co.id/ to rag knowledge base
Agent: Added 'https://www.kayak.co.id/' to the knowledge base (12 chunks indexed).

You: What is kayak.co.id?
Agent: Kayak.co.id is a travel search engine that compares prices across... (Source: https://www.kayak.co.id/)
```

---

## Getting Started

### 1. Prerequisites
- Python 3.8 or higher.

### 2. Setup Virtual Environment
Run the following commands inside the project:

```bash
# Create a virtual environment using uv
uv venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate

# create an empty sqlite db
touch main.db
```

Create the `.env` file, see the file [config.py](./app/config.py) for env used.

### 3. Install Dependencies
```bash
# Synchronize virtual environment with pyproject.toml dependencies
uv sync
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

The project defines the following packages in `pyproject.toml`:

1. **`fastapi`**: The core web framework for building APIs with Python 3.8+ based on standard Python type hints. It provides high performance (on par with NodeJS and Go), rapid coding, and automatic interactive Swagger/ReDoc documentation.
2. **`uvicorn[standard]`**: An ASGI (Asynchronous Server Gateway Interface) web server implementation for Python. FastAPI is built on ASGI standard, and `uvicorn` acts as the server to run the FastAPI application. The `[standard]` extra installs high-performance loop dependencies like `uvloop` and `httptools`.
3. **`pydantic`**: Data validation and settings management using Python type annotations. FastAPI uses Pydantic to parse and validate request JSON payloads, and serialize response objects.
4. **`pytest`**: A robust testing framework for writing clean, readable, and scalable unit tests.
5. **`httpx`**: A next-generation HTTP client for Python. It is used in unit tests alongside FastAPI's `TestClient` to make asynchronous/synchronous HTTP requests to the application. Also used by `rag_ingest_agent` to fetch page content.
6. **`beautifulsoup4`**: HTML parser used by `rag_ingest_agent` to strip tags and extract readable text from scraped pages.
7. **`langchain-text-splitters`**: Splits scraped page text into overlapping chunks before embedding, for the RAG knowledge base.