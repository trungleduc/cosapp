from typing import Any


def noop() -> None:
    """Defines a no-op."""
    return None


def return_arg(arg: Any) -> Any:
    """Returns the object passed as argument."""
    return arg
