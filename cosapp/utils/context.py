class ContextLock:
    """Simple on/off context manager to handle locking mechanisms."""

    def __init__(self):
        self.__active = False

    def __enter__(self):
        self.__active = True

    def __exit__(self, *args, **kwargs):
        self.__active = False

    @property
    def is_active(self) -> bool:
        """bool: Is context activated?"""
        return self.__active
