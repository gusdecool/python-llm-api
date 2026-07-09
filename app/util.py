import uuid

_session_id = str(uuid.uuid4())


def get_session_id() -> str:
    """
    Get the unique session ID for the CLI / process session.
    """
    return _session_id
