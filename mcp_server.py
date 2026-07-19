from datetime import datetime
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from sqlmodel import Session
from app.db import engine, init_db
from app.models import LLMJob
from app.agents import choose_agent, weather_agent, generate_image_agent, marketing_agent, rag_ingest_agent, rag_query_agent


# 1. Initialize the FastMCP Server
mcp = FastMCP("Agent Orchestrator Server")

# Initialize database on module load to ensure schema exists
init_db()


@mcp.tool()
def submit_agent_task(prompt: str) -> str:
    """
    Submits a task to the agent pool (e.g. weather requests, image generation).
    Returns the initial response or asks for more information (HITL).
    """
    with Session(engine) as session:
        # Create a new job in the database
        job = LLMJob(prompt=prompt, status="queue")
        session.add(job)
        session.commit()
        session.refresh(job)

        # Route the job using the agent chooser
        choice = choose_agent(prompt, session=session, user_id="mcp_user")

        if choice.action in ("direct_answer", "unsupported"):
            job.status = "done"
            job.response = choice.direct_response
            job.state = {"agent": choice.action}
            job.responded_at = datetime.utcnow()
            session.add(job)
            session.commit()
            return f"[Job {job.id} - Done]: {choice.direct_response}"

        config = {
            "configurable": {
                "session": session,
                "user_id": "mcp_user"
            }
        }

        try:
            if choice.action == "weather_agent":
                initial_state = {
                    "prompt": prompt,
                    "location": None,
                    "date": None,
                    "missing_fields": [],
                    "next_question": None,
                    "weather_data": None,
                    "final_response": None
                }
                result = weather_agent.invoke(initial_state, config=config)
            elif choice.action == "generate_image_agent":
                initial_state = {
                    "prompt": prompt,
                    "safe_prompt": None,
                    "is_safe": None,
                    "rejection_reason": None,
                    "image_bytes": None,
                    "image_url": None,
                    "final_response": None
                }
                result = generate_image_agent.invoke(initial_state, config=config)
            elif choice.action == "marketing_agent":
                initial_state = {
                    "prompt": prompt,
                    "topic": None,
                    "drafts": None,
                    "approved_option": None,
                    "final_response": None,
                    "next_question": None
                }
                result = marketing_agent.invoke(initial_state, config=config)
            elif choice.action == "rag_ingest_agent":
                initial_state = {
                    "prompt": prompt,
                    "url": None,
                    "title": None,
                    "scraped_text": None,
                    "already_exists": None,
                    "chunk_count": None,
                    "final_response": None
                }
                result = rag_ingest_agent.invoke(initial_state, config=config)
            elif choice.action == "rag_query_agent":
                initial_state = {
                    "prompt": prompt,
                    "query_embedding": None,
                    "retrieved_chunks": None,
                    "final_response": None
                }
                result = rag_query_agent.invoke(initial_state, config=config)
            else:
                # Unsupported action inside this MCP context
                job.status = "error"
                job.response = f"Unsupported agent: {choice.action}"
                session.add(job)
                session.commit()
                return f"[Job {job.id} - Error]: Unsupported agent: {choice.action}"

            # Check if agent requires follow-up input / approval
            if result.get("next_question"):
                job.status = "awaiting_input"
                job.response = result["next_question"]
                status_msg = f"[Job {job.id} - Awaiting Input]: {result['next_question']}"
            else:
                job.status = "done"
                job.response = result.get("final_response") or "Done"
                job.responded_at = datetime.utcnow()
                status_msg = f"[Job {job.id} - Done]: {job.response}"

            job.state = {
                "agent": choice.action,
                **{k: v for k, v in result.items() if k not in ("prompt", "final_response", "next_question", "query_embedding", "scraped_text") and not isinstance(v, bytes)}
            }
            session.add(job)
            session.commit()
            return status_msg

        except Exception as e:
            job.status = "error"
            job.response = f"Failed: {str(e)}"
            session.add(job)
            session.commit()
            return f"[Job {job.id} - Error]: {str(e)}"


@mcp.tool()
def submit_followup_answer(job_id: int, answer: str) -> str:
    """
    Submits a response/answer to a job that is currently status 'awaiting_input' (requires HITL/additional details).
    """
    with Session(engine) as session:
        job = session.get(LLMJob, job_id)
        if not job:
            return f"Error: Job {job_id} not found."

        if job.status != "awaiting_input":
            return f"Error: Job is in status '{job.status}' and does not expect an answer."

        agent_name = job.state.get("agent") if job.state else None
        config = {
            "configurable": {
                "session": session,
                "user_id": "mcp_user"
            }
        }

        try:
            if agent_name == "weather_agent":
                state = {
                    "prompt": answer,
                    "location": job.state.get("location"),
                    "date": job.state.get("date"),
                    "missing_fields": [],
                    "next_question": None,
                    "weather_data": None,
                    "final_response": None
                }
                result = weather_agent.invoke(state, config=config)
            elif agent_name == "marketing_agent":
                state = {
                    "prompt": answer,
                    "topic": job.state.get("topic"),
                    "drafts": job.state.get("drafts"),
                    "approved_option": job.state.get("approved_option"),
                    "final_response": None,
                    "next_question": None
                }
                result = marketing_agent.invoke(state, config=config)
            else:
                return f"Error: Agent '{agent_name}' does not support resume/follow-up in this tool."

            if result.get("next_question"):
                job.status = "awaiting_input"
                job.response = result["next_question"]
                status_msg = f"[Job {job.id} - Awaiting Input]: {result['next_question']}"
            else:
                job.status = "done"
                job.response = result.get("final_response") or "Done"
                job.responded_at = datetime.utcnow()
                status_msg = f"[Job {job.id} - Done]: {job.response}"

            job.state = {
                "agent": agent_name,
                **{k: v for k, v in result.items() if k not in ("prompt", "final_response", "next_question") and not isinstance(v, bytes)}
            }
            session.add(job)
            session.commit()
            return status_msg

        except Exception as e:
            return f"Error running agent: {str(e)}"


if __name__ == "__main__":
    import sys
    # Support running as SSE (http URL) or standard stdio
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        import uvicorn
        from starlette.routing import Route, Mount
        from starlette.responses import Response

        base_app = mcp.sse_app()
        http_base_app = mcp.streamable_http_app()

        original_sse_endpoint = None
        message_app = None
        http_app = None

        for route in base_app.routes:
            if isinstance(route, Route) and route.path == "/sse":
                original_sse_endpoint = route.endpoint
            elif isinstance(route, Mount) and route.path.rstrip("/") == "/messages":
                message_app = route.app

        for route in http_base_app.routes:
            if isinstance(route, Route) and route.path.rstrip("/") == "/mcp":
                http_app = route.endpoint

        if original_sse_endpoint and message_app and http_app:
            class DummyResponse(Response):
                async def __call__(self, scope, receive, send) -> None:
                    pass

            async def custom_sse_endpoint(request):
                if request.method == "GET":
                    return await original_sse_endpoint(request)
                elif request.method == "POST":
                    if request.query_params.get("session_id"):
                        # Forward to standard SSE message handler
                        await message_app(request.scope, request.receive, request._send)
                    else:
                        # Forward to Streamable HTTP handler
                        await http_app(request.scope, request.receive, request._send)
                    return DummyResponse()
                elif request.method == "DELETE":
                    return Response("Session closed", status_code=200)
                return Response("Method not allowed", status_code=405)

            new_routes = []
            for route in base_app.routes:
                if isinstance(route, Route) and route.path == "/sse":
                    new_routes.append(
                        Route(
                            "/sse",
                            endpoint=custom_sse_endpoint,
                            methods=["GET", "POST", "DELETE"]
                        )
                    )
                else:
                    new_routes.append(route)
            
            from starlette.applications import Starlette
            # Combine the lifespans of both apps
            async def combined_lifespan(app):
                async with base_app.router.lifespan_context(base_app):
                    async with http_base_app.router.lifespan_context(http_base_app):
                        yield

            app = Starlette(routes=new_routes, lifespan=combined_lifespan)

        print("Starting custom MCP server with POST/DELETE /sse support on http://localhost:8000/sse")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        mcp.run(transport="stdio")
