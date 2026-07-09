import os
from typing import Optional
from langfuse.langchain import CallbackHandler
from app.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL


_cached_handler: Optional[CallbackHandler] = None


def get_langfuse_handler() -> Optional[CallbackHandler]:
    """
    get Langfuse callback handler.
    this will make it only initialize the handler once.
    """
    global _cached_handler
    if _cached_handler is not None:
        return _cached_handler

    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
        if LANGFUSE_BASE_URL:
            os.environ["LANGFUSE_HOST"] = LANGFUSE_BASE_URL
        _cached_handler = CallbackHandler()
        return _cached_handler
    return None
