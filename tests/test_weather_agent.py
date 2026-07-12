import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch
from sqlmodel import SQLModel, create_engine, Session, StaticPool
from app.main import app
from app.db import get_session
from app.models import LLMJob
from app.agents.agent_chooser import choose_agent, ChooserResult
from app.agents.weather_agent import weather_agent


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
        job1 = LLMJob(prompt="Weather query", status="queue")
        session.add(job1)
        session.commit()
        
    yield
    
    # Drop tables
    SQLModel.metadata.drop_all(engine)


def test_choose_agent_routing():
    from unittest.mock import patch, MagicMock
    from app.agents.agent_chooser import ChooserResult

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    def mock_structured_invoke(input_data, *args, **kwargs):
        prompt_text = ""
        if hasattr(input_data, "to_messages"):
            prompt_text = input_data.to_messages()[-1].content
        elif isinstance(input_data, dict):
            prompt_text = input_data.get("prompt", "")
        else:
            prompt_text = str(input_data)

        if "car hire" in prompt_text or "car rental" in prompt_text:
            return ChooserResult(action="car_hire_agent")
        elif "weather" in prompt_text:
            return ChooserResult(action="weather_agent")
        elif "E=mc2" in prompt_text:
            return ChooserResult(action="direct_answer", direct_response="E=mc2 is energy equals mass times speed of light squared.")
        else:
            return ChooserResult(action="unsupported", direct_response="I can not do that at the moment")

    mock_structured.invoke.side_effect = mock_structured_invoke
    mock_structured.side_effect = mock_structured_invoke

    with patch("app.agents.agent_chooser.get_gemini_2_5_flash_model", return_value=mock_llm):
        r1 = choose_agent("find me car hire in Sydney")
        assert r1.action == "car_hire_agent"
        
        r2 = choose_agent("what is the weather in London")
        assert r2.action == "weather_agent"
        
        r3 = choose_agent("what is E=mc2")
        assert r3.action == "direct_answer"
        assert "E=mc" in r3.direct_response
        
        r4 = choose_agent("order a pizza for me")
        assert r4.action == "unsupported"
        assert r4.direct_response == "I can not do that at the moment"



def test_weather_agent_direct_response():
    mock_choice = ChooserResult(action="weather_agent")
    mock_result = {
        "prompt": "what is the weather in Sydney",
        "location": "Sydney",
        "date": "2026-07-09",
        "missing_fields": [],
        "next_question": None,
        "weather_data": {
            "temp": 18.5,
            "feels_like": 17.0,
            "condition": "clear sky",
            "humidity": 60,
            "wind_speed": 3.5,
            "city": "Sydney"
        },
        "final_response": "The weather in Sydney is clear sky with 18.5°C."
    }
    
    with patch("app.routes.llm_job.choose_agent", return_value=mock_choice), \
         patch("app.routes.llm_job.weather_agent.invoke", return_value=mock_result):
        payload = {"prompt": "what is the weather in Sydney"}
        response = client.post("/llm-job", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "done"
        assert data["response"] == "The weather in Sydney is clear sky with 18.5°C."
        assert data["state"] == {"agent": "weather_agent", "location": "Sydney", "date": "2026-07-09"}


def test_weather_agent_hitl_flow():
    mock_choice = ChooserResult(action="weather_agent")
    
    mock_first_turn = {
        "prompt": "how is the weather today",
        "location": None,
        "date": None,
        "missing_fields": ["location"],
        "next_question": "Which city would you like to check?",
        "weather_data": None,
        "final_response": None
    }
    
    mock_second_turn = {
        "prompt": "London",
        "location": "London",
        "date": "2026-07-09",
        "missing_fields": [],
        "next_question": None,
        "weather_data": {
            "temp": 15.0,
            "feels_like": 14.0,
            "condition": "light rain",
            "humidity": 80,
            "wind_speed": 5.0,
            "city": "London"
        },
        "final_response": "The weather in London is light rain with 15.0°C."
    }
    
    with patch("app.routes.llm_job.choose_agent", return_value=mock_choice), \
         patch("app.routes.llm_job.weather_agent.invoke") as mock_invoke:
        mock_invoke.return_value = mock_first_turn
        
        payload = {"prompt": "how is the weather today"}
        response = client.post("/llm-job", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "awaiting_input"
        assert data["response"] == "Which city would you like to check?"
        job_id = data["id"]
        
        mock_invoke.return_value = mock_second_turn
        response = client.patch(f"/llm-job/{job_id}", json={"answer": "London"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "done"
        assert data["response"] == "The weather in London is light rain with 15.0°C."
        assert data["state"] == {"agent": "weather_agent", "location": "London", "date": "2026-07-09"}
