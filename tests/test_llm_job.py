import os
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session, StaticPool
from app.main import app
from app.db import get_session
from app.models import LLMJob


# Create an in-memory SQLite engine
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

def get_test_session():
    with Session(engine) as session:
        yield session

# Override get_session dependency
app.dependency_overrides[get_session] = get_test_session

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    # Create tables
    SQLModel.metadata.create_all(engine)
    
    # Seed with some data
    with Session(engine) as session:
        job1 = LLMJob(prompt="Job 1 queue", status="queue")
        job2 = LLMJob(prompt="Job 2 processing", status="processing")
        job3 = LLMJob(prompt="Job 3 done", response="Done response", status="done")
        job4 = LLMJob(prompt="Job 4 error", status="error")
        session.add_all([job1, job2, job3, job4])
        session.commit()
        
    yield
    
    # Drop tables
    SQLModel.metadata.drop_all(engine)


def test_create_job():
    payload = {"prompt": "Tell me a joke"}
    response = client.post("/llm-job", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["prompt"] == "Tell me a joke"
    assert data["status"] == "queue"
    assert data["id"] is not None

def test_list_jobs_default_filters():
    response = client.get("/llm-job")
    assert response.status_code == 200
    data = response.json()
    # default filters queue, processing, done (excludes error) -> should return 3
    assert len(data) == 3
    statuses = [job["status"] for job in data]
    assert "error" not in statuses

def test_list_jobs_all_statuses():
    response = client.get("/llm-job?statuses=queue,processing,done,error")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4

def test_list_jobs_pagination():
    response = client.get("/llm-job?limit=2&offset=1&statuses=queue,processing,done,error")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

def test_list_jobs_orderby():
    response = client.get("/llm-job?orderBy=created_at,desc&statuses=queue,processing,done,error")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    # The last seeded item (Job 4) should be first when ordered by created_at desc
    assert data[0]["prompt"] == "Job 4 error"

def test_patch_job_success():
    response = client.patch("/llm-job/1", json={"status": "processing"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    
    response = client.patch("/llm-job/1", json={"status": "done", "response": "Completed!"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "done"
    assert data["response"] == "Completed!"
    assert data["responded_at"] is not None

def test_patch_job_not_found():
    response = client.patch("/llm-job/9999", json={"status": "done"})
    assert response.status_code == 404

def test_patch_job_invalid_payload():
    response = client.patch("/llm-job/1", json={"prompt": "Hack"})
    assert response.status_code == 400
