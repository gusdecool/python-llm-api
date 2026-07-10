from datetime import datetime, timedelta
import re
from typing import Optional
from sqlmodel import Session, select, col
from app.models.llm_memory import LLMMemory
from app.log import get_logger

log = get_logger("agent-memory")


def normalize_key(key: str) -> str:
    """
    Normalize keys for case-insensitive and whitespace-insensitive matching.
    """
    return key.strip().lower().rstrip(".!?")


def get_memory(session: Session, user_id: str, memory_type: str, query_key: str) -> Optional[LLMMemory]:
    """
    Retrieve memory entry if it exists.
    """
    norm_key = normalize_key(query_key)
    statement = select(LLMMemory).where(
        LLMMemory.user_id == user_id,
        LLMMemory.memory_type == memory_type,
        col(LLMMemory.query_key) == norm_key
    )
    result = session.exec(statement).first()
    return result


def save_memory(session: Session, user_id: str, memory_type: str, query_key: str, response_val: str) -> LLMMemory:
    """
    Save or update memory entry.
    """
    norm_key = normalize_key(query_key)
    existing = get_memory(session, user_id, memory_type, norm_key)
    
    if existing:
        existing.response_val = response_val
        existing.created_at = datetime.utcnow()
        session.add(existing)
        session.commit()
        session.refresh(existing)
        log.info(f"Updated memory: {memory_type} -> {norm_key}")
        return existing
    
    new_mem = LLMMemory(
        user_id=user_id,
        memory_type=memory_type,
        query_key=norm_key,
        response_val=response_val,
        created_at=datetime.utcnow()
    )
    session.add(new_mem)
    session.commit()
    session.refresh(new_mem)
    log.info(f"Saved new memory: {memory_type} -> {norm_key}")
    return new_mem


from pydantic import BaseModel, Field
from app.llm_model import get_gemini_2_5_flash_model


class ProfileFact(BaseModel):
    key: str = Field(description="The name of the property or fact in snake_case (e.g., 'name', 'car_preference', 'birthday')")
    value: str = Field(description="The value or detail to remember (e.g., 'Budi', 'cool cars', 'May 5')")


def try_handle_profile(session: Session, user_id: str, prompt: str) -> Optional[str]:
    """
    Intercept identity and profile commands locally to save LLM calls.
    Returns the response text if handled, or None to pass through.
    """
    clean = prompt.strip().lower()
    
    # 1. Check if user is asking to remember something
    is_remember_cmd = (
        clean.startswith("remember") or 
        "remember that" in clean or 
        "keep in mind" in clean or
        clean.startswith("my name is") or
        clean.startswith("i am") or
        clean.startswith("iam")
    )
    
    if is_remember_cmd:

        try:
            llm = get_gemini_2_5_flash_model(temperature=0)
            structured_llm = llm.with_structured_output(ProfileFact)
            
            # Extract key/value using a quick LLM call
            fact = structured_llm.invoke(
                f"Extract the core fact/preference to remember from this user input: '{prompt}'."
            )
            
            if fact and fact.key and fact.value:
                save_memory(session, user_id, "profile", fact.key, fact.value)
                return "Ok."
        except Exception as e:
            log.error(f"Failed to extract memory fact: {str(e)}")
            
        return "Ok."
        
    # 2. Check if user is asking about what we remember about them
    is_about_user = any(w in clean.split() for w in ["i", "me", "my", "myself", "am"])
    is_query_cmd = (
        "who am i" in clean or 
        "who i am" in clean or
        "what is my" in clean or 
        "what do you know about me" in clean or
        "what do you remember" in clean or
        "my name" in clean or
        "do i like" in clean or
        "do you know" in clean
    ) and is_about_user

    
    if is_query_cmd:
        # Fetch all profile memories for this user
        statement = select(LLMMemory).where(
            LLMMemory.user_id == user_id,
            LLMMemory.memory_type == "profile"
        )
        memories = session.exec(statement).all()
        
        if not memories:
            return "I don't know your name or details yet. You can tell me to remember something by saying 'Remember that I like cool cars' or 'Remember my name is Budi'."
            
        # Format the context
        context_lines = []
        for m in memories:
            context_lines.append(f"- {m.query_key}: {m.response_val}")
        context = "\n".join(context_lines)
        
        try:
            llm = get_gemini_2_5_flash_model(temperature=0.2)
            prompt_str = (
                "You are a helpful assistant with access to the user's stored profile details/memories.\n"
                f"Here are the stored facts about the user:\n{context}\n\n"
                f"Answer the user's question: '{prompt}' using these facts. Keep it short and natural."
            )
            response = llm.invoke(prompt_str).content
            return response
        except Exception as e:
            log.error(f"Failed to answer profile query: {str(e)}")
            
    return None




def check_prompt_cache(session: Session, user_id: str, prompt: str) -> Optional[dict]:
    """
    Check if the user has asked this exact/similar prompt recently.
    """
    norm_prompt = normalize_key(prompt)
    
    # Check general direct answer cache first
    mem = get_memory(session, user_id, "direct_answer", norm_prompt)
    if mem:
        log.info(f"Cache hit for direct answer: {norm_prompt}")
        return {"action": "direct_answer", "response": mem.response_val}
        
    # Check image cache
    mem = get_memory(session, user_id, "image", norm_prompt)
    if mem:
        log.info(f"Cache hit for image: {norm_prompt}")
        return {"action": "generate_image_agent", "response": mem.response_val}
        
    # Check weather cache with 15-minute TTL
    mem = get_memory(session, user_id, "weather", norm_prompt)
    if mem:
        age = datetime.utcnow() - mem.created_at
        if age < timedelta(minutes=15):
            log.info(f"Cache hit for weather (age: {age}): {norm_prompt}")
            return {"action": "weather_agent", "response": mem.response_val}
        else:
            log.info(f"Weather cache expired for: {norm_prompt}")
            
    return None
