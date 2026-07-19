# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python multi-agent orchestration system built on LangChain/LangGraph, exposed via three interfaces (REST API, CLI, MCP server) that all share the same agent/routing/DB layer. Observability is via Langfuse; models are accessed through LiteLLM (currently Gemini 2.5 Flash).

## Commands

```bash
# Setup
uv venv venv
source venv/bin/activate
touch main.db          # create empty sqlite db
uv sync                # install deps from pyproject.toml

# Run REST API (http://127.0.0.1:8000, docs at /docs)
uvicorn app.main:app --reload

# Run interactive CLI
python -m cli

# Run MCP server (stdio, for Claude Desktop / Cursor)
./venv/bin/python mcp_server.py

# Run MCP server (SSE/HTTP, http://localhost:8000/sse)
python -m mcp_server sse

# Run all tests
PYTHONPATH=. pytest

# Run a single test file / test
PYTHONPATH=. pytest tests/test_weather_agent.py
PYTHONPATH=. pytest tests/test_weather_agent.py::test_choose_agent_routing
```

Use `./venv/bin/python` / `./venv/bin/pytest` explicitly when the venv isn't activated (per `.agents/AGENTS.md`).

Tests set required env vars (mock API keys, `DATABASE_URL=sqlite:///main_test.db`) in `tests/conftest.py`, and use an in-memory SQLite engine + `app.dependency_overrides` per test module — no `.env` needed to run the suite.

## Architecture

### Three entrypoints, one core

`app/main.py` (FastAPI), `cli.py`, and `mcp_server.py` are thin wrappers around the same core: `app.agents.choose_agent` for routing + LangGraph agent graphs (`car_hire_agent`, `weather_agent`, `generate_image_agent`, `marketing_agent`, `rag_ingest_agent`, `rag_query_agent`) for execution. All three entrypoints duplicate the same per-agent "build initial state / invoke / handle `next_question`" dispatch logic — when changing an agent's state shape or adding an agent, update all three call sites plus `app/agents/__init__.py`.

### Request flow (REST, mirrored by CLI/MCP)

1. `POST /llm-job` (`app/routes/llm_job.py`) creates an `LLMJob` row (`app/models/llm_job.py`), status `queue`.
2. `choose_agent()` (`app/agents/agent_chooser.py`) routes the prompt. Before calling the LLM it short-circuits through `app/agents/agent_memory.py`:
   - `try_handle_profile` — handles "remember X" / "who am I" locally.
   - `check_prompt_cache` — exact-match cache for prior `direct_answer`/`image` responses, and a 15-minute TTL cache for `weather`.
   - Only if neither hits does it call Gemini with structured output to pick an action (`car_hire_agent` / `weather_agent` / `generate_image_agent` / `marketing_agent` / `rag_ingest_agent` / `rag_query_agent` / `direct_answer` / `unsupported`). Before this call it also fetches the most recent `RagDocument` URLs (`get_known_rag_sources`) and includes them in the router's system prompt, so questions about previously-ingested sites route to `rag_query_agent` instead of `direct_answer`/`unsupported`.
3. Direct/unsupported answers finish the job immediately (status `done`). Otherwise the matching LangGraph agent is invoked with `config={"configurable": {"session": ..., "user_id": ...}}` so agent nodes can read/write `LLMMemory` (`app/models/llm_memory.py`) via the DB session.
4. If the agent graph returns `next_question` (a field on its `TypedDict` state), the job goes to status `awaiting_input` and the response is the follow-up question; the client resumes it via `PATCH /llm-job/{id}` with `{"answer": ...}`, which re-invokes the same agent (`agent_name` read back from `job.state["agent"]`) with prior state fields merged in. Otherwise the job is `done` (or `error` if the agent raised `AppException`/an exception).

### LangGraph agents (`app/agents/`)

Each agent (`car_hire_agent.py`, `weather_agent.py`, `generate_image_agent.py`, `marketing_agent.py`) follows the same shape: a `TypedDict` state, Pydantic schemas for structured LLM output, node functions built as `ChatPromptTemplate | llm.with_structured_output(...)`, and a compiled `StateGraph` with conditional edges for the human-in-the-loop (`missing_fields`/`next_question`) or approval (`marketing_agent`'s `approved_option`) pattern. Every LLM-calling node passes `get_langfuse_handler()` as a callback and tags calls with `langfuse_tags: [agent_name, node_name]` — follow this pattern for new nodes/agents so traces stay attributable in Langfuse.

`rag_ingest_agent.py`/`rag_query_agent.py` are a linear (no HITL) pair implementing the URL-RAG flow from `specs/url-rag.md`: `rag_ingest_agent` regex-extracts a URL from the prompt (no LLM call — deterministic and cheaper than structured extraction for this), skips re-scraping if the URL is already in `RagDocument`, otherwise scrapes with `httpx` + `beautifulsoup4`, splits with `langchain_text_splitters.RecursiveCharacterTextSplitter`, and embeds/persists chunks via `GoogleGenerativeAIEmbeddings` (`app.config.GEMINI_EMBEDDING_MODEL`, reuses `GEMINI_API_KEY`). `rag_query_agent` embeds the question, ranks all `RagChunk` rows by cosine similarity (plain Python, no vector store — retrieval is brute-force, fine at this scale but not built to scale past a small KB), and synthesizes an answer from the top matches, falling back to a fixed "don't know" response below the similarity threshold rather than hallucinating.

`agent_chooser.py` is the router; new agents must be registered in its system prompt (action name + routing description), re-exported from `app/agents/__init__.py`, and wired into the dispatch blocks in `app/main.py`/`routes/llm_job.py`, `cli.py`, and `mcp_server.py`.

### Persistence

- `app/db.py`: single SQLModel `engine` (SQLite by default), `init_db()` creates tables and seeds two demo `LLMJob` rows if empty, `get_session()` is the FastAPI dependency.
- `LLMJob` (`app/models/llm_job.py`): the job/task record — `prompt`, `response`, `status` (`queue`/`awaiting_input`/`done`/`error`), `state` (JSON blob holding the active agent name + its in-progress fields for resumption).
- `LLMMemory` (`app/models/llm_memory.py`): generic key/value cache keyed by `(user_id, memory_type, query_key)` where `query_key` is normalized via `normalize_key()` (lowercased, trimmed, trailing punctuation stripped). `memory_type` values in use: `profile`, `direct_answer`, `image`, `weather`, `weather_data`.
- `RagDocument`/`RagChunk` (`app/models/rag.py`): the URL-RAG knowledge base — one `RagDocument` per ingested URL, one `RagChunk` per text chunk with its embedding stored as JSON (`Column(JSON)`, same pattern as `LLMJob.state`). Unlike `LLMMemory`, this isn't per-`user_id` — all ingested content is shared/global across users.

### Model/config layer

- `app/config.py` loads all env vars (via `.env`) as module-level constants — always add new secrets/settings here rather than reading `os.environ` elsewhere.
- `app/llm_model.py` centralizes model construction (currently only `get_gemini_2_5_flash_model`, using `ChatLiteLLM`); `car_hire_agent.py` still constructs `ChatLiteLLM` directly instead of using this helper (inconsistency to be aware of, not necessarily to copy).
- `app/langfuse.py` lazily builds and caches a single `CallbackHandler`; `get_langfuse_handler()` returns `None` if keys aren't configured, and call sites are expected to handle that (`config = {...} if handler else {}`).
- `app/util.py` provides a single process-lifetime `get_session_id()` used as the Langfuse session id.

## Repository Conventions

From `.agents/AGENTS.md` (binding for all coding agents in this repo):

- **Style**: PEP 8, `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, 4-space indents, two blank lines around top-level defs/classes, one blank line between methods.
- **Imports**: absolute imports rooted at the project (`from app.config import X`, never bare `from config import X`); expose package members via `__init__.py` `__all__` where applicable (see `app/models/__init__.py`).
- **SQLite**: never use the `COMMENT` keyword in SQL; always manage DB sessions via context managers or FastAPI `Depends`.
- **FastAPI/Pydantic**: use `lifespan` (not `@app.on_event`), Pydantic v2's `.model_dump()` (not `.dict()`), `typing.Annotated` for route dependencies.
- Read config from `app.config`, not `os.environ`, directly.
- All prompts sent to an LLM must go through Langfuse observability (pass the callback handler + `langfuse_tags`/`langfuse_session_id` metadata, matching existing node patterns).
- `TODO` comments mark code intentionally left alone — do not change unless explicitly asked.

## Domain-specific skills

`.agents/skills/` contains reference skills for LangChain/LangGraph/Langfuse/deep-agents patterns (fundamentals, RAG, middleware, persistence, human-in-the-loop, orchestration, etc.). Consult the relevant skill under this directory before making non-trivial changes to agent graphs, memory/persistence, or Langfuse instrumentation.
