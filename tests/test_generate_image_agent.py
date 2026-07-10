import json
import pytest
from unittest.mock import patch, MagicMock
from sqlmodel import SQLModel, create_engine, Session, StaticPool
from fastapi.testclient import TestClient

from app.main import app
from app.db import get_session
from app.models import LLMJob
from app.agents.agent_chooser import choose_agent
from app.agents.generate_image_agent import generate_image_agent, SafetyCheckResult

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
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)


def test_choose_agent_routing_image():
    with patch("app.agents.agent_chooser.GEMINI_API_KEY", "mock_key"):
        # We need to mock the LLM's invocation to return a ChooserResult for generate_image_agent
        mock_result = MagicMock()
        mock_result.action = "generate_image_agent"
        mock_result.direct_response = None

        with patch("langchain_core.prompts.ChatPromptTemplate.from_messages") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_result
            mock_prompt.return_value.__or__.return_value = mock_chain

            res = choose_agent("generate an image of a red cat")
            assert res.action == "generate_image_agent"


def test_generate_image_agent_success():
    # Mock parameters
    mock_safety_result = SafetyCheckResult(
        is_safe=True,
        rejection_reason=None,
        safe_prompt="A beautiful red cat sitting on a desk"
    )

    mock_image_response = MagicMock()
    mock_image_data = MagicMock()
    mock_image_data.url = "http://temp.url/image.png"
    mock_image_response.data = [mock_image_data]

    # Mock safety check chain
    mock_safety_chain = MagicMock()
    mock_safety_chain.invoke.return_value = mock_safety_result
    mock_safety_chain.return_value = mock_safety_result

    # Mock S3 Client
    mock_s3 = MagicMock()

    with patch("app.agents.generate_image_agent.get_gemini_2_5_flash_model") as mock_llm_factory, \
         patch("litellm.image_generation", return_value=mock_image_response) as mock_litellm, \
         patch("httpx.get") as mock_httpx_get, \
         patch("boto3.client", return_value=mock_s3), \
         patch("app.agents.generate_image_agent.AWS_ACCESS_KEY_ID", "key"), \
         patch("app.agents.generate_image_agent.AWS_SECRET_ACCESS_KEY", "secret"), \
         patch("app.agents.generate_image_agent.AWS_REGION", "ap-southeast-1"), \
         patch("app.agents.generate_image_agent.AWS_S3_BUCKET", "bungamata-public"):

        # Mock LLM structure chain
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_safety_chain
        mock_llm_factory.return_value = mock_llm

        # Mock image download response
        mock_http_resp = MagicMock()
        mock_http_resp.content = b"fake-image-bytes"
        mock_httpx_get.return_value = mock_http_resp

        # Invoke generate_image_agent
        state = {
            "prompt": "generate a red cat",
            "safe_prompt": None,
            "is_safe": None,
            "rejection_reason": None,
            "image_bytes": None,
            "image_url": None,
            "final_response": None
        }

        result = generate_image_agent.invoke(state)

        # Assertions
        assert result["is_safe"] is True
        assert result["image_url"].startswith("https://bungamata-public.s3.ap-southeast-1.amazonaws.com/")
        
        # Verify JSON content
        final_resp = json.loads(result["final_response"])
        assert final_resp["status"] == "success"
        assert final_resp["url"] == result["image_url"]
        
        # Verify S3 client was called
        mock_s3.put_object.assert_called_once()


def test_generate_image_agent_safety_rejection():
    mock_safety_result = SafetyCheckResult(
        is_safe=False,
        rejection_reason="Adult content detected",
        safe_prompt=None
    )

    mock_safety_chain = MagicMock()
    mock_safety_chain.invoke.return_value = mock_safety_result
    mock_safety_chain.return_value = mock_safety_result

    with patch("app.agents.generate_image_agent.get_gemini_2_5_flash_model") as mock_llm_factory, \
         patch("litellm.image_generation") as mock_litellm:

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_safety_chain
        mock_llm_factory.return_value = mock_llm

        state = {
            "prompt": "some unsafe prompt",
            "safe_prompt": None,
            "is_safe": None,
            "rejection_reason": None,
            "image_bytes": None,
            "image_url": None,
            "final_response": None
        }

        result = generate_image_agent.invoke(state)

        # Assertions
        assert result["is_safe"] is False
        assert result["rejection_reason"] == "Adult content detected"
        assert result["image_url"] is None
        
        final_resp = json.loads(result["final_response"])
        assert final_resp["status"] == "rejected"
        assert final_resp["reason"] == "Adult content detected"
        
        # Ensure image generation was NOT called
        mock_litellm.assert_not_called()
