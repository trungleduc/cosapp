import pytest
from cosapp.utils.deprecation import deprecated


def test_deprecated_function_warns():
    @deprecated("use another function")
    def old_func():
        return 42

    with pytest.warns(DeprecationWarning, match="use another function"):
        result = old_func()
    
    assert result == 42


def test_deprecated_method_warns():
    class ClassWithDeprecatedMethod:
        @deprecated("use another method")
        def old_method(self):
            return "ok"
    
    obj = ClassWithDeprecatedMethod()

    with pytest.warns(DeprecationWarning, match="use another method"):
        result = obj.old_method()
    
    assert result == "ok"


def test_deprecated_with_redirect():
    """Test redirection of a deprecated function to a new function."""
    def new_func(n: int):
        return n + 1

    @deprecated(redirect=new_func)
    def old_func(n: int): ...

    with pytest.warns(DeprecationWarning, match="use `new_func` instead"):
        result = old_func(1)
    
    assert result == 2


def test_deprecated_method_redirection():
    """Test redirection of a deprecated method to a new method."""
    class ClassWithRedirectedMethod:
        def new_method(self, n: int):
            """Great method that adds 1 to argument n."""
            return n + 1

        @deprecated(redirect=new_method)
        def old_method(self, n: int): ...

    obj = ClassWithRedirectedMethod()

    with pytest.warns(DeprecationWarning, match="use `new_method` instead"):
        result = obj.old_method(3)
    
    assert result == 4
    assert obj.old_method.__doc__ == "\n".join([
        "Great method that adds 1 to argument n.",
        "",
        "Deprecated; use `new_method` instead.",
    ])


def test_deprecated_method_redirection_with_message():
    """Test redirection of a deprecated method to a new method."""
    class ClassWithRedirectedMethod:
        def new_method(self, n: int):
            """Great method that adds 1 to argument n."""
            return n + 1

        @deprecated("too bad!", redirect=new_method)
        def old_method(self, n: int): ...

    obj = ClassWithRedirectedMethod()

    with pytest.warns(DeprecationWarning, match="`old_method` is deprecated; too bad!; use `new_method` instead."):
        result = obj.old_method(3)
    
    assert result == 4
    assert obj.old_method.__doc__ == "\n".join([
        "Great method that adds 1 to argument n.",
        "",
        "Deprecated; too bad!; use `new_method` instead.",
    ])
