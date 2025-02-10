from typing import Any


class Proxy:
    """Simple proxy base class"""

    def __init__(self, wrapped: Any):
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)
