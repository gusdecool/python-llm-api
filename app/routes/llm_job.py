from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select, col
from pydantic import BaseModel, Field
from langfuse.langchain import CallbackHandler
from app.db import get_session
from app.models import LLMJob
from app.agents import car_hire_agent
from app.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL


router = APIRouter(prefix="/llm-job", tags=["LLM Job"])


class LLMJobCreate(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt for the LLM job")


class LLMJobUpdate(BaseModel):
    status: Optional[str] = Field(None, description="Status of the job")
    response: Optional[str] = Field(None, description="Response from the LLM")
    answer: Optional[str] = Field(None, description="User's answer to the follow-up question")


def get_langfuse_handler() -> Optional[CallbackHandler]:
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        import os
        os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
        if LANGFUSE_BASE_URL:
            os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL
        return CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY)
    return None


@router.post("", response_model=LLMJob, status_code=status.HTTP_201_CREATED)
def create_job(payload: LLMJobCreate, session: Session = Depends(get_session)):
    # Initialize the job in database
    job = LLMJob(prompt=payload.prompt, status="queue")
    session.add(job)
    session.commit()
    session.refresh(job)

    # Invoke agent with Langfuse observability callbacks
    handler = get_langfuse_handler()
    config = {"callbacks": [handler]} if handler else {}
    
    initial_state = {
        "prompt": payload.prompt,
        "location": None,
        "start_date": None,
        "end_date": None,
        "missing_fields": [],
        "next_question": None,
        "scraped_deals": None,
        "final_response": None
    }
    
    try:
        result = car_hire_agent.invoke(initial_state, config=config)
        
        if result.get("next_question"):
            job.status = "awaiting_input"
            job.response = result["next_question"]
        else:
            job.status = "done"
            job.response = result.get("final_response") or "Done"
            job.responded_at = datetime.utcnow()
            
        job.state = {
            "location": result.get("location"),
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date")
        }
    except Exception as e:
        job.status = "error"
        job.response = f"Agent failed: {str(e)}"
        job.responded_at = datetime.utcnow()

    session.add(job)
    session.commit()
    session.refresh(job)
    return job


@router.get("", response_model=List[LLMJob])
def list_jobs(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    statuses: str = Query(default="queue,processing,done"),
    orderBy: Optional[str] = Query(default=None, description="Format: field,asc|desc"),
    session: Session = Depends(get_session)
):
    # Parse statuses from comma separated string
    status_list = [s.strip() for s in statuses.split(",") if s.strip()]
    
    # Build query
    query = select(LLMJob)
    if status_list:
        query = query.where(col(LLMJob.status).in_(status_list))
    
    # Order by logic
    if orderBy:
        parts = orderBy.split(",")
        field_name = parts[0].strip()
        direction = parts[1].strip().lower() if len(parts) > 1 else "asc"
        
        field = getattr(LLMJob, field_name, None)
        if field is not None:
            if direction == "desc":
                query = query.order_by(col(field).desc())
            else:
                query = query.order_by(col(field).asc())
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid orderBy field: {field_name}"
            )
            
    # Limit and offset
    query = query.offset(offset).limit(limit)
    
    return session.exec(query).all()


@router.patch("/{id}", response_model=LLMJob)
def update_job(
    id: int,
    payload: LLMJobUpdate,
    session: Session = Depends(get_session)
):
    job = session.get(LLMJob, id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LLM Job with ID {id} not found"
        )
    
    # Handle answer logic which triggers the agent run
    if payload.answer is not None:
        handler = get_langfuse_handler()
        config = {"callbacks": [handler]} if handler else {}
        
        agent_state = {
            "prompt": payload.answer,
            "location": job.state.get("location") if job.state else None,
            "start_date": job.state.get("start_date") if job.state else None,
            "end_date": job.state.get("end_date") if job.state else None,
            "missing_fields": [],
            "next_question": None,
            "scraped_deals": None,
            "final_response": None
        }
        
        try:
            result = car_hire_agent.invoke(agent_state, config=config)
            
            if result.get("next_question"):
                job.status = "awaiting_input"
                job.response = result["next_question"]
            else:
                job.status = "done"
                job.response = result.get("final_response") or "Done"
                job.responded_at = datetime.utcnow()
                
            job.state = {
                "location": result.get("location"),
                "start_date": result.get("start_date"),
                "end_date": result.get("end_date")
            }
        except Exception as e:
            job.status = "error"
            job.response = f"Agent failed: {str(e)}"
            job.responded_at = datetime.utcnow()
            
    else:
        # standard update logic if answer is not provided
        update_data = payload.model_dump(exclude_unset=True)
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one field (status, response, or answer) must be provided for update"
            )
            
        for key, value in update_data.items():
            if key != "answer":
                setattr(job, key, value)
            
        if "response" in update_data or update_data.get("status") in ("done", "error"):
            job.responded_at = datetime.utcnow()
        
    session.add(job)
    session.commit()
    session.refresh(job)
    return job
