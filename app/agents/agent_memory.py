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


def try_handle_profile(session: Session, user_id: str, prompt: str) -> Optional[str]:
    """
    Intercept identity and profile commands locally to save LLM calls.
    Returns the response text if handled, or None to pass through.
    """
    clean = prompt.strip().lower()
    
    save_match = re.search(r"\bremember\s+(?:that\s+)?(?:my\s+name\s+is|i\s*am|i'm|iam)\s+([a-zA-Z0-9_\-\s]+)", clean)
    if not save_match:
        # Match assertions but not query/question forms like "who am i"
        if not re.search(r"\bwho\s+(?:am\s+i|i\s*am)\b", clean):
            save_match = re.search(r"\b(?:my\s+name\s+is|i\s*am|i'm|iam)\s+([a-zA-Z0-9_\-\s]+)", clean)

        
    if save_match:
        name = save_match.group(1).strip().title()
        save_memory(session, user_id, "profile", "name", name)
        return "Ok."
        
    # 2. Check if user is asking who they are
    if (re.search(r"\bwho\s+(?:am\s+i|i\s*am)\b", clean) or 
            re.search(r"\bwhat\s+is\s+my\s+name\b", clean) or 
            re.search(r"\bdo\s+you\s+know\s+my\s+name\b", clean)):
        mem = get_memory(session, user_id, "profile", "name")
        if mem:
            return f"You're {mem.response_val}"
        return "I don't know your name yet. You can tell me by saying 'Remember my name is Budi'."
        
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
