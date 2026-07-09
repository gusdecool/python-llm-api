class AppException(Exception):
    """
    Generic exception that happen during runtime
    """

    def __init__(self, message: str = "Sorry, unexpected error happened. Please try again or contact us"):
        self.message = message

