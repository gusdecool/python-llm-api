import pytest
from unittest.mock import patch, MagicMock
from sqlmodel import SQLModel, create_engine, Session, StaticPool, select

from app.models import RagDocument, RagChunk
from app.agents.rag_ingest_agent import rag_ingest_agent
from app.agents.rag_query_agent import rag_query_agent, NO_KNOWLEDGE_RESPONSE

# Create an in-memory SQLite engine
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)


MOCK_HTML = """
<html>
<head><title>Example Domain</title></head>
<body>
<script>var trackingId = "abc";</script>
<h1>Example Domain</h1>
<p>This domain is for use in illustrative examples.</p>
</body>
</html>
"""


def fake_embed_documents(texts, task_type=None):
    return [[1.0, 0.0] for _ in texts]


def test_rag_ingest_agent_success():
    mock_http_response = MagicMock()
    mock_http_response.text = MOCK_HTML
    mock_http_response.raise_for_status.return_value = None

    with Session(engine) as session, \
         patch("httpx.get", return_value=mock_http_response), \
         patch("app.agents.rag_ingest_agent.GoogleGenerativeAIEmbeddings") as mock_embeddings_cls:

        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.side_effect = fake_embed_documents
        mock_embeddings_cls.return_value = mock_embeddings_instance

        state = {
            "prompt": "Add https://www.example.com/ to rag knowledge base",
            "url": None,
            "title": None,
            "scraped_text": None,
            "already_exists": None,
            "chunk_count": None,
            "final_response": None
        }
        config = {"configurable": {"session": session, "user_id": "default_user"}}

        result = rag_ingest_agent.invoke(state, config=config)

        assert result["url"] == "https://www.example.com/"
        assert result["chunk_count"] == 1
        assert "Added" in result["final_response"]

        documents = session.exec(select(RagDocument)).all()
        chunks = session.exec(select(RagChunk)).all()
        assert len(documents) == 1
        assert documents[0].url == "https://www.example.com/"
        assert len(chunks) == 1
        assert chunks[0].embedding == [1.0, 0.0]


def test_rag_ingest_agent_already_exists():
    with Session(engine) as session:
        session.add(RagDocument(url="https://www.example.com/", title="Example", char_count=100))
        session.commit()

    with Session(engine) as session, patch("httpx.get") as mock_get:
        state = {
            "prompt": "Add https://www.example.com/ to rag knowledge base",
            "url": None,
            "title": None,
            "scraped_text": None,
            "already_exists": None,
            "chunk_count": None,
            "final_response": None
        }
        config = {"configurable": {"session": session, "user_id": "default_user"}}

        result = rag_ingest_agent.invoke(state, config=config)

        assert "already in the knowledge base" in result["final_response"]
        mock_get.assert_not_called()


def test_rag_ingest_agent_no_url():
    with Session(engine) as session:
        state = {
            "prompt": "Add kayak.co.id to rag knowledge base",
            "url": None,
            "title": None,
            "scraped_text": None,
            "already_exists": None,
            "chunk_count": None,
            "final_response": None
        }
        config = {"configurable": {"session": session, "user_id": "default_user"}}

        result = rag_ingest_agent.invoke(state, config=config)

        assert "valid http(s) URL" in result["final_response"]


def test_rag_query_agent_no_knowledge():
    with Session(engine) as session, \
         patch("app.agents.rag_query_agent.GoogleGenerativeAIEmbeddings") as mock_embeddings_cls:

        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.return_value = [1.0, 0.0]
        mock_embeddings_cls.return_value = mock_embeddings_instance

        state = {
            "prompt": "What is kayak.co.id?",
            "query_embedding": None,
            "retrieved_chunks": None,
            "final_response": None
        }
        config = {"configurable": {"session": session, "user_id": "default_user"}}

        result = rag_query_agent.invoke(state, config=config)

        assert result["final_response"] == NO_KNOWLEDGE_RESPONSE


def test_rag_query_agent_with_match():
    with Session(engine) as session:
        doc = RagDocument(url="https://www.kayak.co.id/", title="Kayak", char_count=500)
        session.add(doc)
        session.commit()
        session.refresh(doc)
        session.add(RagChunk(document_id=doc.id, chunk_index=0, content="Kayak compares flight and car prices.", embedding=[1.0, 0.0]))
        session.commit()

    with Session(engine) as session, \
         patch("app.agents.rag_query_agent.GoogleGenerativeAIEmbeddings") as mock_embeddings_cls, \
         patch("app.agents.rag_query_agent.get_gemini_2_5_flash_model") as mock_llm_factory:

        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_query.return_value = [1.0, 0.0]
        mock_embeddings_cls.return_value = mock_embeddings_instance

        mock_response = MagicMock()
        mock_response.content = "Kayak compares prices across providers. Source: https://www.kayak.co.id/"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm.return_value = mock_response
        mock_llm_factory.return_value = mock_llm

        state = {
            "prompt": "What is kayak.co.id?",
            "query_embedding": None,
            "retrieved_chunks": None,
            "final_response": None
        }
        config = {"configurable": {"session": session, "user_id": "default_user"}}

        result = rag_query_agent.invoke(state, config=config)

        assert len(result["retrieved_chunks"]) == 1
        assert result["retrieved_chunks"][0]["url"] == "https://www.kayak.co.id/"
        assert result["final_response"] == mock_response.content
