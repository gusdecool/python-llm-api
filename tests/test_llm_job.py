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

@pytest.fixture(autouse=True)
def override_dependencies():
    app.dependency_overrides[get_session] = get_test_session
    yield
    app.dependency_overrides.clear()

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
    from unittest.mock import patch
    from app.agents.agent_chooser import ChooserResult
    
    mock_choice = ChooserResult(action="car_hire_agent")
    mock_agent_result = {
        "prompt": "Tell me a joke",
        "location": None,
        "start_date": None,
        "end_date": None,
        "missing_fields": ["location"],
        "next_question": "Could you please provide the missing details: location?",
        "scraped_deals": None,
        "final_response": None
    }

    
    with patch("app.routes.llm_job.choose_agent", return_value=mock_choice), \
         patch("app.routes.llm_job.car_hire_agent.invoke", return_value=mock_agent_result):
        payload = {"prompt": "Tell me a joke"}
        response = client.post("/llm-job", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["prompt"] == "Tell me a joke"
        assert data["status"] == "awaiting_input"
        assert "missing" in data["response"].lower()
        assert data["id"] is not None


def test_car_hire_agent_flow_with_mock():
    # Mocking the agent to simulate complete details and success path
    from unittest.mock import patch
    from app.agents.agent_chooser import ChooserResult
    
    mock_choice = ChooserResult(action="car_hire_agent")
    mock_result = {
        "prompt": "find me car hire in Brisbane for 17 July to 21 July",
        "location": "Brisbane",
        "start_date": "2026-07-17",
        "end_date": "2026-07-21",
        "missing_fields": [],
        "next_question": None,
        "scraped_deals": [
            {
                "provider": "Kayak.com (via Hertz)",
                "location": "Brisbane",
                "car_model": "Toyota Corolla",
                "price_per_day": 45.0,
                "total_price": 180.0,
                "url": "mock_url"
            }
        ],
        "final_response": "Here is the best deal for Brisbane: Toyota Corolla at $45/day."
    }
    
    with patch("app.routes.llm_job.choose_agent", return_value=mock_choice), \
         patch("app.routes.llm_job.car_hire_agent.invoke", return_value=mock_result):
        payload = {"prompt": "find me car hire in Brisbane for 17 July to 21 July"}
        response = client.post("/llm-job", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "done"
        assert "Toyota Corolla" in data["response"]
        assert data["state"] == {"agent": "car_hire_agent", "location": "Brisbane", "start_date": "2026-07-17", "end_date": "2026-07-21"}


def test_car_hire_agent_hitl_flow_with_mock():
    from unittest.mock import patch
    from app.agents.agent_chooser import ChooserResult
    
    mock_choice = ChooserResult(action="car_hire_agent")
    
    # 1. First turn: User asks for car hire but missing location
    mock_first_turn = {
        "prompt": "find me car hire for 17 July to 21 July",
        "location": None,
        "start_date": "2026-07-17",
        "end_date": "2026-07-21",
        "missing_fields": ["location"],
        "next_question": "Could you please provide the location?",
        "scraped_deals": None,
        "final_response": None
    }
    
    # 2. Second turn: User provides location -> completes search
    mock_second_turn = {
        "prompt": "Brisbane",
        "location": "Brisbane",
        "start_date": "2026-07-17",
        "end_date": "2026-07-21",
        "missing_fields": [],
        "next_question": None,
        "scraped_deals": [],
        "final_response": "Found deals for Brisbane!"
    }
    
    with patch("app.routes.llm_job.choose_agent", return_value=mock_choice), \
         patch("app.routes.llm_job.car_hire_agent.invoke") as mock_invoke:
        mock_invoke.return_value = mock_first_turn
        payload = {"prompt": "find me car hire for 17 July to 21 July"}
        response = client.post("/llm-job", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "awaiting_input"
        assert data["response"] == "Could you please provide the location?"
        job_id = data["id"]
        
        # Simulating user response to patch
        mock_invoke.return_value = mock_second_turn
        response = client.patch(f"/llm-job/{job_id}", json={"answer": "Brisbane"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert data["response"] == "Found deals for Brisbane!"
        assert data["state"] == {"agent": "car_hire_agent", "location": "Brisbane", "start_date": "2026-07-17", "end_date": "2026-07-21"}



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
