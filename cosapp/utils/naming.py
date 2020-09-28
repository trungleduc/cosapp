import re
from typing import Any, List, NoReturn
from cosapp.utils.helpers import check_arg


def has_time(expression: Any) -> bool:
    """Checks if an expression contains 't'"""
    return re.search(r"\bt\b", str(expression)) is not None


class NameChecker:
    """Class handling admissible names, through regular expression filtering"""
    def __init__(self,
        pattern = r"^[A-Za-z][\w]*$",
        message = "Name must start with a letter, and contain only alphanumerics and '_'"
        ):
        self.__error_message = lambda name: None  # type: Callable[[str], str]
        self.__message = ""  # type: str
        self.__pattern = None  # type: re.Pattern
        self.pattern = pattern
        self.message = message

    @classmethod
    def reserved(cls) -> List[str]:
        """List of reserved names"""
        return ["t", "time"]

    @property
    def pattern(self) -> "re.Pattern":
        return self.__pattern

    @pattern.setter
    def pattern(self, pattern: str) -> NoReturn:
        self.__pattern = re.compile(pattern)

    @property
    def message(self) -> str:
        """Returns a human-readable message, transcripted from regexp rule"""
        return self.__message

    @message.setter
    def message(self, message: str) -> NoReturn:
        check_arg(message, "message", str)
        self.__message = message
        if len(self.__message) > 0:
            self.__error_message = lambda name: f"{self.__message}; got {name!r}."
        else:
            self.__error_message = lambda name: f"Invalid name {name!r}"

    def is_valid(self, name: str) -> bool:
        """Method indicating whether or not a name is valid, under the current rule"""
        try:
            self(name)
        except:
            return False
        else:
            return True

    def __call__(self, name) -> str:
        """Returns `name` if valid; otherwise, raises an exception"""
        check_arg(name, "name", str)
        message = None
        reserved = self.reserved()
        if name in reserved:
            message = f"Names {reserved} are reserved"
        elif self.__pattern.match(name) is None:
            message = self.__error_message(name)
        if message is not None:
            raise ValueError(message)
        return name
