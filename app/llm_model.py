"""
LLM Model Provider
LLM Model Provider is a module that provides a centralized model chooser and selecting keys
"""
from app.app_exception import AppException
from app.config import GEMINI_API_KEY
from langchain_litellm import ChatLiteLLM


def get_gemini_2_5_flash_model(**kwargs) -> ChatLiteLLM:
    """
    Gemini 2.5 flash.
    Support: text, images, video, audio
    Throw AppException if failed
    """
    if not GEMINI_API_KEY:
        raise AppException("please set GEMINI_API_KEY in env")
        
    return ChatLiteLLM(model="gemini/gemini-2.5-flash", api_key=GEMINI_API_KEY, **kwargs)
    
