import re
from typing import TypedDict, Optional, Dict, Any
import httpx
from bs4 import BeautifulSoup
from sqlmodel import select
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from app.app_exception import AppException
from app.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL
from app.models import RagDocument, RagChunk
from app.log import get_logger

log = get_logger("rag-ingest-agent")

URL_REGEX = re.compile(r"https?://[^\s<>\"')\]]+")
REQUEST_TIMEOUT = 10.0
MAX_CONTENT_CHARS = 300_000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class RagIngestState(TypedDict):
    prompt: str
    url: Optional[str]
    title: Optional[str]
    scraped_text: Optional[str]
    already_exists: Optional[bool]
    chunk_count: Optional[int]
    final_response: Optional[str]


def _get_session(config: Optional[RunnableConfig]):
    if config and isinstance(config, dict) and config.get("configurable"):
        return config["configurable"].get("session")
    return None


# Node 1: Extract the URL from the prompt (plain regex, no LLM call needed)
def extract_url(state: RagIngestState) -> Dict[str, Any]:
    match = URL_REGEX.search(state["prompt"])
    if not match:
        return {
            "url": None,
            "final_response": "Please provide a valid http(s) URL to add to the knowledge base."
        }
    url = match.group(0).rstrip(".,;:!?")
    if not url.lower().startswith(("http://", "https://")):
        return {
            "url": None,
            "final_response": "Please provide a valid http(s) URL to add to the knowledge base."
        }
    return {"url": url}


# Node 2: Skip re-scraping if the URL is already indexed
def check_existing(state: RagIngestState, config: RunnableConfig = None) -> Dict[str, Any]:
    session = _get_session(config)
    if session:
        existing = session.exec(select(RagDocument).where(RagDocument.url == state["url"])).first()
        if existing:
            log.info(f"URL already indexed, skipping scrape: {state['url']}")
            return {
                "already_exists": True,
                "final_response": f"'{state['url']}' is already in the knowledge base ({existing.char_count} characters indexed)."
            }
    return {"already_exists": False}


# Node 3: Scrape the URL and extract readable text
def scrape_url(state: RagIngestState) -> Dict[str, Any]:
    url = state["url"]
    log.info(f"Scraping {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentOrchestratorBot/1.0)"}
        response = httpx.get(url, timeout=REQUEST_TIMEOUT, headers=headers, follow_redirects=True)
        response.raise_for_status()
    except Exception as e:
        raise AppException(f"failed to fetch {url}: {str(e)}")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    cleaned = "\n".join(lines)[:MAX_CONTENT_CHARS]

    if not cleaned:
        raise AppException(f"no readable content found at {url}")

    return {"scraped_text": cleaned, "title": title}


# Node 4: Split into chunks, embed, and persist
def chunk_and_embed(state: RagIngestState, config: RunnableConfig = None) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise AppException("please set GEMINI_API_KEY in env")

    session = _get_session(config)
    if not session:
        raise AppException("no database session available to store knowledge base content")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text(state["scraped_text"])

    if not chunks:
        raise AppException(f"no content to index from {state['url']}")

    embeddings_model = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    try:
        vectors = embeddings_model.embed_documents(chunks, task_type="RETRIEVAL_DOCUMENT")
    except Exception as e:
        raise AppException(f"failed to embed content: {str(e)}")

    document = RagDocument(url=state["url"], title=state.get("title"), char_count=len(state["scraped_text"]))
    session.add(document)
    session.commit()
    session.refresh(document)

    for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
        session.add(RagChunk(document_id=document.id, chunk_index=idx, content=chunk_text, embedding=vector))
    session.commit()

    log.info(f"Indexed {len(chunks)} chunks for {state['url']}")
    return {"chunk_count": len(chunks)}


# Node 5: Format the final response
def format_response(state: RagIngestState) -> Dict[str, Any]:
    response = (
        f"Added '{state['url']}' to the knowledge base "
        f"({state.get('chunk_count', 0)} chunks indexed)."
    )
    return {"final_response": response}


# Conditional routing
def route_after_extract(state: RagIngestState) -> str:
    return "check_existing" if state.get("url") else "no_url"


def route_after_check(state: RagIngestState) -> str:
    return "exists" if state.get("already_exists") else "scrape"


# Construct the graph
workflow = StateGraph(RagIngestState)

workflow.add_node("extract_url", extract_url)
workflow.add_node("check_existing", check_existing)
workflow.add_node("scrape_url", scrape_url)
workflow.add_node("chunk_and_embed", chunk_and_embed)
workflow.add_node("format_response", format_response)

workflow.set_entry_point("extract_url")

workflow.add_conditional_edges(
    "extract_url",
    route_after_extract,
    {
        "no_url": END,
        "check_existing": "check_existing"
    }
)

workflow.add_conditional_edges(
    "check_existing",
    route_after_check,
    {
        "exists": END,
        "scrape": "scrape_url"
    }
)

workflow.add_edge("scrape_url", "chunk_and_embed")
workflow.add_edge("chunk_and_embed", "format_response")
workflow.add_edge("format_response", END)

# Compile graph
rag_ingest_agent = workflow.compile()
