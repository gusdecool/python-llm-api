# standarize the log format
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger("llm-api")


def get_logger(name: str | None = None) -> logging.Logger:
    """
    get logger with standarized format, default name to "llm-api"
    """

    return logging.getLogger(name or "llm-api")