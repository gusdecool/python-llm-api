# Project Styling and Coding Rules

Every coding agent working in this repository must strictly adhere to the following Python and FastAPI standards:

## 1. Code Style and Conventions (PEP 8)
- **Functions & Variables**: Use `snake_case` (e.g., `init_db()`, `get_db_conn()`).
- **Classes**: Use `PascalCase` (e.g., `CarHireState`, `LLMJob`).
- **Constants**: Use `UPPER_SNAKE_CASE` (e.g., `DATABASE_URL`).
- **Indentation**: Use exactly 4 spaces per indentation level. Do not use tabs.
- **Blank Lines**: 
  - Surround top-level functions and classes with **two blank lines**.
  - Surround class methods with **one blank line**.

## 2. Imports and Packaging (Go-style Root Imports)
- Always prefer **absolute imports** relative to the project root (e.g., `from app.config import DATABASE_URL` instead of `from config import DATABASE_URL`).
- Expose public package components (like models) via `__init__.py` using the `__all__` exports list, allowing clean imports (e.g., `from app.models import LLMJob`).

## 3. Database (SQLite)
- Do not use the `COMMENT` keyword in SQLite SQL statements, as it is unsupported. Use standard SQL comments (`-- comment`).
- Manage all database connections and sessions using Python's **context managers** (`with` statement) or FastAPI's dependency injection (`Depends`) to guarantee automatic session/connection closing.

## 4. Modern FastAPI & Pydantic Conventions
- Use the `lifespan` event handler instead of the deprecated `@app.on_event("startup")` or `"shutdown"`.
- Use Pydantic v2's `.model_dump()` instead of the deprecated `.dict()`.
- Use Python's built-in `typing.Annotated` for route dependencies and metadata (e.g. `conn: Annotated[Session, Depends(get_session)]`).

## 5. Modern Python Conventions
- Use config from `app.config` instead of `os.environ` directly to map all the config required
- All prompt must be send to Langfuse for observavibility.
- Use `./venv/bin` to execute Python related command