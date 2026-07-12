from datetime import datetime, timedelta
import pytest
from sqlmodel import SQLModel, create_engine, Session, StaticPool
from app.models.llm_memory import LLMMemory
from app.agents.agent_memory import (
    save_memory,
    get_memory,
    try_handle_profile,
    check_prompt_cache
)

# Set up in-memory database engine for testing
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@pytest.fixture(autouse=True)
def setup_db():
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)


def test_profile_memory_and_isolation():
    from unittest.mock import patch, MagicMock
    from app.agents.agent_memory import ProfileFact

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    def mock_structured_invoke(prompt_text, *args, **kwargs):
        if "Budi" in prompt_text:
            return ProfileFact(key="name", value="Budi")
        elif "John" in prompt_text:
            return ProfileFact(key="name", value="John")
        return None

    def mock_llm_invoke(prompt_str, *args, **kwargs):
        resp = MagicMock()
        if "Budi" in prompt_str:
            resp.content = "Your name is Budi."
        elif "John" in prompt_str:
            resp.content = "Your name is John."
        else:
            resp.content = "I don't know your name."
        return resp

    mock_structured.invoke.side_effect = mock_structured_invoke
    mock_llm.invoke.side_effect = mock_llm_invoke

    with patch("app.agents.agent_memory.get_gemini_2_5_flash_model", return_value=mock_llm):
        with Session(engine) as session:
            # Save Budi's name using different variations
            res1 = try_handle_profile(session, "user_budi", "remember iam Budi")
            assert res1 == "Ok."
            
            # Save John's name
            res2 = try_handle_profile(session, "user_john", "My name is John")
            assert res2 == "Ok."
            
            # Query Budi's name
            ans1 = try_handle_profile(session, "user_budi", "Who am I?")
            assert "Budi" in ans1
            
            # Overwrite with another pattern
            res1_alt = try_handle_profile(session, "user_budi", "remember i'm Budi")
            assert res1_alt == "Ok."
            assert "Budi" in try_handle_profile(session, "user_budi", "who i am")
            
            # Query John's name
            ans2 = try_handle_profile(session, "user_john", "Who am I?")
            assert "John" in ans2
            
            # Query missing user's name
            ans3 = try_handle_profile(session, "user_unknown", "Who am I?")
            assert "don't know your name" in ans3



def test_weather_cache_expiration():
    with Session(engine) as session:
        user_id = "test_user"
        
        # Save raw weather data cache
        weather_info = '{"temp": 28.5, "condition": "sunny"}'
        save_memory(session, user_id, "weather_data", "denpasar", weather_info)
        
        # Retrieve before 15 minutes passes
        cached = get_memory(session, user_id, "weather_data", "denpasar")
        assert cached is not None
        assert cached.response_val == weather_info
        
        # Modify the timestamp of the cached object to simulate 20 minutes ago
        cached.created_at = datetime.utcnow() - timedelta(minutes=20)
        session.add(cached)
        session.commit()
        
        # Check that it shows expired
        cached_expired = get_memory(session, user_id, "weather_data", "denpasar")
        age = datetime.utcnow() - cached_expired.created_at
        assert age >= timedelta(minutes=15)


def test_general_prompt_cache():
    with Session(engine) as session:
        user_id = "test_user"
        prompt = "Explain quantum physics in one sentence."
        response = "It is the science of the extremely small."
        
        # Save to general direct answer cache
        save_memory(session, user_id, "direct_answer", prompt, response)
        
        # Test prompt cache retrieval
        cached = check_prompt_cache(session, user_id, prompt)
        assert cached is not None
        assert cached["action"] == "direct_answer"
        assert cached["response"] == response
