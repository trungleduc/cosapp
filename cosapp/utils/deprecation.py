import functools
import warnings
from typing import Callable, Optional


def deprecated(message="", redirect: Optional[Callable] = None):
    """Decorator to mark functions or methods as deprecated.

    Parameters
    ----------
    - message (str): Deprecation message to display.
    - redirect (callable, optional): Function called instead of the decorated function.
    """
    if message:
        message = f"; {message}"
    if redirect:
        message += f"; use `{redirect.__name__}` instead."

    def decorator(func: Callable):
        if redirect is None:
            f_called = func
        else:
            f_called = redirect
            doc_extra = f"Deprecated{message}"
            if (source_doc := redirect.__doc__):
                source_doc = source_doc.strip()
                spacing = "\n" if source_doc.endswith("\n") else "\n\n"
                func.__doc__ = f"{source_doc}{spacing}{doc_extra}"
            else:
                func.__doc__ = doc_extra
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"Function `{func.__name__}` is deprecated" + message,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return f_called(*args, **kwargs)
        
        return wrapper
    
    return decorator
