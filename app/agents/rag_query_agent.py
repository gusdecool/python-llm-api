from typing import TypedDict, Optional, List, Dict, Any
from sqlmodel import select
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
from app.app_exception import AppException
from app.llm_model import get_gemini_2_5_flash_model
from app.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL
from app.models import RagChunk, RagDocument
from app.log import get_logger
from app.langfuse import get_langfuse_handler
from app.util import get_session_id

log = get_logger("rag-query-agent")

TOP_K = 5
MIN_SIMILARITY = 0.6
NO_KNOWLEDGE_RESPONSE = "I don't have information about that in my knowledge base yet. Try adding a relevant URL first."


class RagQueryState(TypedDict):
    prompt: str
    query_embedding: Optional[List[float]]
    retrieved_chunks: Optional[List[Dict[str, Any]]]
    final_response: Optional[str]


def _get_session(config: Optional[RunnableConfig]):
    if config and isinstance(config, dict) and config.get("configurable"):
        return config["configurable"].get("session")
    return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Node 1: Embed the user's question
def embed_query(state: RagQueryState) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise AppException("please set GEMINI_API_KEY in env")

    embeddings_model = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
    try:
        vector = embeddings_model.embed_query(state["prompt"], task_type="RETRIEVAL_QUERY")
    except Exception as e:
        raise AppException(f"failed to embed query: {str(e)}")

    return {"query_embedding": vector}


# Node 2: Rank stored chunks by cosine similarity to the query
def retrieve_chunks(state: RagQueryState, config: RunnableConfig = None) -> Dict[str, Any]:
    session = _get_session(config)
    if not session:
        return {"retrieved_chunks": []}

    query_vector = state["query_embedding"]
    all_chunks = session.exec(select(RagChunk)).all()

    scored = []
    for chunk in all_chunks:
        if not chunk.embedding:
            continue
        score = _cosine_similarity(query_vector, chunk.embedding)
        if score >= MIN_SIMILARITY:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, chunk in scored[:TOP_K]:
        document = session.get(RagDocument, chunk.document_id)
        results.append({
            "content": chunk.content,
            "url": document.url if document else None,
            "score": score
        })

    log.info(f"Retrieved {len(results)} relevant chunks for query")
    return {"retrieved_chunks": results}


# Node 3: Synthesize the answer from retrieved context
def synthesize_answer(state: RagQueryState) -> Dict[str, Any]:
    retrieved = state.get("retrieved_chunks") or []
    if not retrieved:
        return {"final_response": NO_KNOWLEDGE_RESPONSE}

    context = "\n\n".join(f"Source: {chunk['url']}\n{chunk['content']}" for chunk in retrieved)

    llm = get_gemini_2_5_flash_model(temperature=0.2)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant answering questions using ONLY the provided knowledge base context.\n"
            "If the context does not contain the answer, say you don't have that information.\n"
            "Cite the source URL(s) you used."
        )),
        ("user", "Context:\n{context}\n\nQuestion: {prompt}")
    ])
    chain = prompt_template | llm

    try:
        handler = get_langfuse_handler()
        config = {
            "callbacks": [handler],
            "metadata": {
                "langfuse_session_id": get_session_id(),
                "langfuse_tags": ["rag_query_agent", "synthesize_answer"]
            }
        } if handler else {}
        response = chain.invoke({"context": context, "prompt": state["prompt"]}, config=config).content
    except Exception as e:
        raise AppException(f"failed to synthesize answer: {str(e)}")

    return {"final_response": response}


# Construct the graph
workflow = StateGraph(RagQueryState)

workflow.add_node("embed_query", embed_query)
workflow.add_node("retrieve_chunks", retrieve_chunks)
workflow.add_node("synthesize_answer", synthesize_answer)

workflow.set_entry_point("embed_query")
workflow.add_edge("embed_query", "retrieve_chunks")
workflow.add_edge("retrieve_chunks", "synthesize_answer")
workflow.add_edge("synthesize_answer", END)

# Compile graph
rag_query_agent = workflow.compile()
