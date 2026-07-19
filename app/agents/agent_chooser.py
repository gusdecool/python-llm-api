from app.llm_model import get_gemini_2_5_flash_model
from app.config import GEMINI_API_KEY
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm import ChatLiteLLM
from app.log import get_logger
from app.langfuse import get_langfuse_handler
from app.util import get_session_id
from sqlmodel import Session, select, col
from app.agents.agent_memory import try_handle_profile, check_prompt_cache, save_memory
from app.models import RagDocument


class ChooserResult(BaseModel):
    action: str = Field(
        ...,
        description="The action to take: 'car_hire_agent', 'weather_agent', 'generate_image_agent', 'marketing_agent', 'rag_ingest_agent', 'rag_query_agent', 'direct_answer', or 'unsupported'"
    )
    direct_response: Optional[str] = Field(
        None,
        description="The response content if action is 'direct_answer' or 'unsupported'"
    )


log = get_logger('agent-chooser')

KNOWN_SOURCES_LIMIT = 20


def get_known_rag_sources(session: Session, limit: int = KNOWN_SOURCES_LIMIT) -> List[str]:
    """
    Fetch a bounded list of previously ingested URLs so the router can recognize
    questions that relate to content already added to the RAG knowledge base.
    """
    statement = select(RagDocument).order_by(col(RagDocument.fetched_at).desc()).limit(limit)
    documents = session.exec(statement).all()
    return [doc.url for doc in documents]


def choose_agent(prompt: str, session: Optional[Session] = None, user_id: str = "default_user") -> ChooserResult:
    """
    Choose AI agent based on prompt.
    If the request is can be answered directly, use 'direct_answer' action.
    Otherwise if the request is something that we do not have tool for it, use 'unsupported' action.

    Args:
        prompt: User prompt
        session: SQLModel database session
        user_id: Unique user identifier for isolation

    Returns:
        ChooserResult with action and direct_response
    """
    if session:
        # 1. Intercept profile query / updates locally (no LLM call)
        profile_response = try_handle_profile(session, user_id, prompt)
        if profile_response is not None:
            log.info("Handled profile memory request locally")
            return ChooserResult(action="direct_answer", direct_response=profile_response)

        # 2. Check general caches (exact prompt matches for weather, image, direct_answer)
        cached = check_prompt_cache(session, user_id, prompt)
        if cached:
            log.info(f"Handled prompt cache request locally for {cached['action']}")
            return ChooserResult(action=cached["action"], direct_response=cached["response"])

    log.info('chosing agent')
    llm = get_gemini_2_5_flash_model(temperature=0)

    structured_llm = llm.with_structured_output(ChooserResult)

    known_sources = get_known_rag_sources(session) if session else []
    known_sources_text = (
        "\n".join(f"- {url}" for url in known_sources)
        if known_sources else "(none yet)"
    )

    # TODO use abstraction to build system prompt and agent discovery from app.agents.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an routing agent. Analyze the user prompt and decide the action:\n"
            "- If the user asks for car hire, car rental, or vehicle booking, set action to 'car_hire_agent'.\n"
            "- If the user asks about the weather, temperature, rain, or weather forecast, set action to 'weather_agent'.\n"
            "- If the user asks to generate, draw, paint, create, or design an image, picture, photo, or graphic, set action to 'generate_image_agent'.\n"
            "- If the user asks to draft, generate, or write a marketing campaign, ad copy, marketing copy, or Facebook/Instagram/Google ad, set action to 'marketing_agent'.\n"
            "- If the user asks to add, scrape, index, or learn a URL/website into the knowledge base (e.g. 'add https://... to rag knowledge base'), set action to 'rag_ingest_agent'.\n"
            "- If the user asks a question that relates to one of the Known Knowledge Base Sources below (by domain, brand, or topic, even if not an exact match), or explicitly asks to look something up 'from the knowledge base' / 'from what you scraped', set action to 'rag_query_agent'.\n"
            "- If the user asks to remember a fact/preference about them, or asks what you know/remember about them (e.g. 'who am i', 'do you know my name'), set action to 'direct_answer' and write a helpful response in direct_response.\n"
            "- If the user asks a general knowledge question that you can answer directly (e.g. 'what is E=mc2'), set action to 'direct_answer' and write the answer in direct_response.\n"
            "- If the user asks for anything else that requires a tool we do not have (e.g. booking a flight, ordering food, writing code), set action to 'unsupported' and set direct_response to exactly 'I can not do that at the moment'.\n\n"
            "Known Knowledge Base Sources (URLs already added via rag_ingest_agent):\n"
            "{known_sources}"
        )),
        ("user", "{prompt}")
    ])

    chain = prompt_template | structured_llm
    try:
        handler = get_langfuse_handler()
        config = {
            "callbacks": [handler],
            "metadata": {
                "langfuse_session_id": get_session_id(),
                "langfuse_tags": ["agent_chooser", "choose_agent"]
            }
        } if handler else {}

        result = chain.invoke({"prompt": prompt, "known_sources": known_sources_text}, config=config)

        if result.action == "unsupported":
            result.direct_response = "I can not do that at the moment"
        
        # Save direct answers to cache if session is provided
        if session and result.action == "direct_answer" and result.direct_response:
            save_memory(session, user_id, "direct_answer", prompt, result.direct_response)

        return result
    except Exception:
        return ChooserResult(action="unsupported", direct_response="I can not do that at the moment")

