from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select, col
from app.db import get_session
from app.models import LLMJob
from pydantic import BaseModel, Field

router = APIRouter(prefix="/llm-job", tags=["LLM Job"])

class LLMJobCreate(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt for the LLM job")

class LLMJobUpdate(BaseModel):
    status: Optional[str] = Field(None, description="Status of the job")
    response: Optional[str] = Field(None, description="Response from the LLM")

@router.post("", response_model=LLMJob, status_code=status.HTTP_201_CREATED)
def create_job(payload: LLMJobCreate, session: Session = Depends(get_session)):
    job = LLMJob(prompt=payload.prompt, status="queue")
    session.add(job)


    # use Langfuse 

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
    
    update_data = payload.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field (status or response) must be provided for update"
        )
        
    for key, value in update_data.items():
        setattr(job, key, value)
        
    if "response" in update_data or update_data.get("status") in ("done", "error"):
        job.responded_at = datetime.utcnow()
        
    session.add(job)
    session.commit()
    session.refresh(job)
    return job
