import numpy
from typing import Any, Optional, Tuple, List, NamedTuple
from collections.abc import Collection

from .helpers import check_arg
from .naming import natural_varname


class MaskedVarInfo(NamedTuple):
    basename: str
    selector: str = ""
    mask: Optional[numpy.ndarray] = None

    @property
    def fullname(self) -> bool:
        return f"{self.basename}{self.selector}"


def find_selector(expression: str) -> Tuple[str, str]:
    """Decompose a string expression into `basename` and `selector`,
    where `selector` is a suitable mask expression for an array.

    Parameters:
    -----------
    expression [str]: the expression to be parsed.
    
    Returns:
    --------
    baseline, selector [str, str]
    """
    expression = expression.strip()
    left_bracket, right_bracket = brackets = tuple("[]")
    nl = nr = il = 0

    basename, selector = expression, ""  # default

    if expression.endswith(right_bracket):
        i = len(expression)
        for char in reversed(expression):
            i -= 1
            if char == right_bracket:
                nr += 1
            elif char == left_bracket:
                nl += 1
                if nl == nr:
                    prev_char = expression[i - 1] if i > 0 else None
                    if prev_char not in brackets:
                        il = i
                        break
    
    elif expression.endswith(left_bracket):
        nl, nr = 1, 0
    
    balanced = (nl == nr)
    if not balanced:
        raise ValueError(f"Bracket mismatch in {expression!s}")
    if il > 0 or expression.startswith(left_bracket):
        basename = expression[:il]
        selector = expression[il:]

    return basename, selector


# TODO should we reintegrate it as Boundary method as it is now the only used place
def get_indices(system: "cosapp.base.System", name: str) -> MaskedVarInfo:
    """Decompose a variable specification into its base name and mask.

    Parameters
    ----------
    system : System
        System to which variable belongs
    name : str
        Variable specification (variable name + optional array mask, if required)

    Returns
    -------
    MaskedVarInfo (named tuple):
        - basename [str]: variable name
        - selector [str]: array selector
        - mask [numpy.ndarray[bool]]: mask (if array) or `None`
    """
    check_arg(name, 'name', str)
    name = natural_varname(name)

    def check_eval(attribute: str) -> Any:
        try:
            value = eval(f"s.{attribute}", {}, {"s": system})
        except AttributeError as error:
            error.args = (f"{attribute!r} is not known in {system.name}",)
            raise
        except Exception as error:
            error.args = (f"Can't evaluate {attribute!r} in {system.name}",)
            raise
        else:
            return value

    try:
        basename, selector = find_selector(name)
    except ValueError as error:
        raise SyntaxError(error)

    value = check_eval(basename)
    mask = None

    if selector:
        # Check value is an array
        if not (
            isinstance(value, numpy.ndarray)
            and numpy.issubdtype(value.dtype, numpy.number)
            and value.size > 1
        ):
            raise TypeError(
                f"Only non-empty numpy arrays can be partially selected; got '{system.name}.{name}'."
            )
        mask = numpy.zeros_like(value, dtype=bool)
        # Set mask from selector string
        try:
            exec(f"mask{selector} = True", {}, {"mask": mask})
        except (SyntaxError, IndexError) as error:
            varname = f"{system.name}.{basename}"
            error.args = (
                f"Invalid selector {selector!r} for variable {varname!r}: {error!s}",
            )
            raise
    
    elif isinstance(value, Collection) and not isinstance(value, str):
        mask = numpy.ones_like(value, dtype=bool)

    return MaskedVarInfo(basename, selector, mask)


def multi_split(expression: str, separators: List[str]) -> Tuple[List[str], List[str]]:
    """Extension of `str.split`, accounting for more than one split separators.
    
    Parameters:
    -----------
    - expression [str]:
        Expression to be split.
    - separators [List[str]]:
        List of separators.
    
    Returns:
    --------
    - expressions [List[str]]:
        List of n split expressions.
    - separators [List[str]]:
        Sequence of (n - 1) separators between split expressions.

    Examples:
    ---------
    >>> multi_split('a+b-c-d+e', list('+-'))
    ['a', 'b', 'c', 'd', 'e'], ['+', '-', '-', '+']
    """
    expressions = [expression]
    sep_list = []

    for separator in set(separators):
        new_list = []
        shift = 0
        for i, expression in enumerate(expressions):
            sublist = expression.split(separator)
            n_hits = len(sublist) - 1
            new_list.extend(sublist)
            if n_hits > 0:
                j = i + shift
                shift += n_hits
                sep_list = sep_list[:j] + [separator] * n_hits + sep_list[j:]
        expressions = list(map(str.strip, new_list))

    return expressions, sep_list
