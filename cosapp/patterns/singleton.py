class Singleton(type):
    """
    Metaclass for singleton pattern.

    Reference
    ---------
    https://refactoring.guru/design-patterns/singleton

    Examples
    --------

    >>> class MyClass(metaclass=Singleton):
    >>>     pass
    >>> 
    >>> m1 = MyClass()
    >>> m2 = MyClass()
    >>> assert m1 is m2
    """
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.__instance = None

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__call__(*args, **kwargs)
        return cls.__instance
