"""
Module handling the execution of code provided as string by the user.

This code is inspired from the OpenMDAO module openmdao.components.exec_comp.
"""
from __future__ import annotations
import re
import numpy
import ast
from enum import Enum
from numbers import Number
from typing import (
    Any, Optional, Callable,
    TYPE_CHECKING,
)
from cosapp.ports.port import BasePort
from types import CodeType

from cosapp.utils.state_io import object__getstate__

if TYPE_CHECKING:
    from cosapp.systems import System


class AstVisitor(ast.NodeTransformer):
    """Visitor of AST python of the string expression to evaluate.

    Convert initial string expression with 'object.attributes' patterns to
    'object_attributes' relatives allowing to handle them as unique variables
    without exploring the whole object tree during value update.

    Parameters
    ----------
    - context : cosapp.systems.System
        CoSApp System to which variables in expression can be retrieved.
    """

    def __init__(self, context: System):
        """Initialization parameters:
        ----------
        - context : cosapp.systems.System
            CoSApp System to which variables in expression can be retrieved.
        """
        self._context = context
        self._vars = set()
        self._expr_vars = {}

    def map_attributes(self, attr: str, key: str = None) -> None:
        """Generic method to reach attribute getter mapping."""
        key = attr if key is None else key
        self._expr_vars[key] = (self._context, attr)
        self._vars.add(attr)

        if attr in self._context.name2variable:
            var_ref = self._context.name2variable[attr]
            if isinstance(var_ref.mapping, BasePort):
                self._expr_vars[key] = (var_ref.mapping, var_ref.key)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute | ast.Name:
        """Visit a `Attribute` and return a concatenate ast.Name if possible."""
        # Determine the longest object.attributes path
        attr_str = ast.unparse(node)
        func_attr = ""
        max_iter = len(attr_str.split("."))
        i = 0
        while i < max_iter and not hasattr(self._context, attr_str):
            split_attr = attr_str.rsplit(".", maxsplit=1)
            attr_str = split_attr[0]
            func_attr = ".".join(split_attr[1:])
            i += 1

        # Collect variables from expression
        if attr_str and i != max_iter:
            key = attr_str.replace(".", "_")
            self.map_attributes(attr_str, key)

            # Return custom node
            if func_attr:
                ast_func = ast.parse(f"{key}.{func_attr}").body[0].value
                return ast_func
            else:
                return ast.Name(id=key, ctx=ast.Load())

        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Visit a `Name` and return a concatenate ast.Name if possible."""
        if hasattr(self._context, node.id):
            self.map_attributes(node.id)
        return node
    
class EvalString:
    """Create a executable statement using an expression string.

    The following functions are available for use in expression:

    =========================  ====================================
    Function                   Description
    =========================  ====================================
    abs(x)                     Absolute value of x
    acos(x)                    Inverse cosine of x
    acosh(x)                   Inverse hyperbolic cosine of x
    arange(start, stop, step)  Array creation
    arccos(x)                  Inverse cosine of x
    arccosh(x)                 Inverse hyperbolic cosine of x
    arcsin(x)                  Inverse sine of x
    arcsinh(x)                 Inverse hyperbolic sine of x
    arctan(x)                  Inverse tangent of x
    asin(x)                    Inverse sine of x
    asinh(x)                   Inverse hyperbolic sine of x
    atan(x)                    Inverse tangent of x
    cos(x)                     Cosine of x
    cosh(x)                    Hyperbolic cosine of x
    cross(x, y)                Cross product of arrays x and y
    dot(x, y)                  Dot-product of x and y
    e                          Euler's number
    erf(x)                     Error function
    erfc(x)                    Complementary error function
    exp(x)                     Exponential function
    expm1(x)                   exp(x) - 1
    factorial(x)               Factorial of all numbers in x
    fmax(x, y)                 Element-wise maximum of x and y
    fmin(x, y)                 Element-wise minimum of x and y
    inner(x, y)                Inner product of arrays x and y
    isinf(x)                   Element-wise detection of numpy.inf
    isnan(x)                   Element-wise detection of numpy.nan
    kron(x, y)                 Kronecker product of arrays x and y
    linspace(x, y, N)          Numpy linear spaced array creation
    log(x)                     Natural logarithm of x
    log10(x)                   Base-10 logarithm of x
    log1p(x)                   log(1+x)
    matmul(x, y)               Matrix multiplication of x and y
    maximum(x, y)              Element-wise maximum of x and y
    minimum(x, y)              Element-wise minimum of x and y
    ones(N)                    Create an array of ones
    outer(x, y)                Outer product of x and y
    pi                         Pi
    power(x, y)                Element-wise x**y
    prod(x)                    The product of all elements in x
    sin(x)                     Sine of x
    sinh(x)                    Hyperbolic sine of x
    sum(x)                     The sum of all elements in x
    round(x)                   Round all elements in x
    tan(x)                     Tangent of x
    tanh(x)                    Hyperbolic tangent of x
    tensordot(x, y)            Tensor dot product of x and y
    zeros(N)                   Create an array of zeros
    =========================  ====================================

    Full list is returned by `EvalString.available_symbols()`

    Parameters
    ----------
    expression : str
        Interpretable Python statement. In addition to standard Python operators,
        a subset of numpy and scipy functions is supported.
    context : cosapp.core.module.Module or cosapp.core.numerics.basics.Residue
        System or Residue defining the local context in which the statement will be executed

    Notes
    -----
    The context is used to include the `System` ports, children, inwards and outwards.
    If the context is a `Residue`, the reference value from the residue is added as
    `residue_reference` variable in the execution context.
    """

    # this dict will act as the global scope when we eval our expressions
    __globals = {}  # type: dict[str, Any]

    @classmethod
    def available_symbols(cls) -> dict[str, Any]:
        """
        List of available symbols (constants and functions) in current execution context.

        Returns
        -------
        dict[str, Any]
            Mapping of available symbols by their name.
        """
        mapping = cls.__globals

        if mapping:
            return mapping
        
        def add_symbols(
            module: object,
            names: Optional[list[str] | tuple[str, str]] = None,
        ) -> None:
            """
            Map attribute names from the given module into the global dict.

            Parameters
            ----------
            mod : object
                Module to check.
            names : iter of str, optional
                If supplied, only map attrs that match the given names
            """
            # nonlocal mapping
            if names is None:
                names = dir(module)
            for name in names:
                if isinstance(name, (list, tuple)):
                    name, alias = name
                else:
                    alias = name
                if not name.startswith("_"):
                    mapping[alias] = mapping[name] = getattr(module, name)

        add_symbols(numpy,
            names=[
                # Numpy types
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "float32",
                "float64",
                "complex64",
                "complex128",
                # Array creation
                "array",
                "asarray",
                "arange",
                "concatenate",
                "ones",
                "zeros",
                "full",
                "full_like",
                "linspace",
                "logspace",
                # Constants
                "e",
                "pi",
                "inf",
                "isinf",
                "isnan",  # Logic
                "log",
                "log10",
                "log1p",
                "power",
                # Math & array operations
                "abs",
                "sqrt",
                "cbrt",
                "exp",
                "expm1",
                "fmax",
                "fmin",
                "maximum",
                "minimum",
                "round",
                "sum",
                "prod",
                "tensordot",
                "where",
                # Linear algebra
                "matmul",
                "cross",
                "outer",
                "inner",
                "kron",
                "dot",
                # Trigo
                "sin",
                "cos",
                "tan",
                ("arcsin", "asin"),
                ("arccos", "acos"),
                ("arctan", "atan"),
                ("arctan2", "atan2"),
                "degrees",
                "radians",
                # Hyperbolic trigo
                "sinh",
                "cosh",
                "tanh",
                ("arcsinh", "asinh"),
                ("arccosh", "acosh"),
                ("arctanh", "atanh"),
            ],
        )
        add_symbols(numpy.linalg, names=["norm"])

        # if scipy is available, add few specials functions
        try:
            import scipy.special
        except ImportError:
            pass
        else:
            add_symbols(scipy.special, names=["factorial", "erf", "erfc"])

        return mapping

    class _EvalConstant:
        def __init__(self, value):
            self._value = value
        
        def __call__(self):
            return self._value

        def __reduce_ex__(self, _):
            return type(self), (self._value, ), {}
  
    class _EvalNotConstant:
        def __init__(self, context: System, source_code: str, code: CodeType, expr_vars: dict[str, tuple]):
            self._context = context
            self._source_code = source_code
            self._code = code
            self._expr_vars = expr_vars
            self._global_dict = EvalString.available_symbols()

        def __call__(self):
            return eval(self._code, self._global_dict, self.locals())
    
        @classmethod
        def _builder(cls, context: System, source_code: str, expr_vars: dict[str, tuple]):
            code = compile(source_code, "<string>", "eval")
            return cls(context, source_code, code, expr_vars)
        
        def __reduce_ex__(self, _) -> tuple[Callable, tuple, dict]:
            return self._builder, (self._context, self._source_code, self._expr_vars), {}

        def locals(self) -> dict[str, Any]:
            """dict[str, Any]: Context attributes required to evaluate the string expression."""
            return {
                key: getattr(obj, attr_name)
                for key, (obj, attr_name) in self._expr_vars.items()
            }

    def __init__(self, expression: Any, context: System) -> None:
        """Class constructor.

        Compiles an expression, and checks that it is evaluable within a given context.
        """
        from cosapp.systems import System
        if not isinstance(context, System):
            cname = type(context).__name__
            raise TypeError(
                f"Object of type {cname!r} is not a valid context to evaluate expression '{expression}'."
            )
        self.__context = context

        self.__str = self.string(expression)

        if len(self.__str) == 0:
            raise ValueError("Can't evaluate empty expressions")

        #  Visit expression from its AST 
        ast_from_str = ast.parse(self.__str).body[0]
        if not isinstance(ast_from_str, (ast.Expression, ast.Expr)):
            raise SyntaxError(f"Expression {self.__str} must be in a correct format.")
        ast_visitor = AstVisitor(context)
        ast_visited = ast.fix_missing_locations(ast_visitor.visit(ast_from_str.value))
        code = compile(ast.Expression(ast_visited), "<string>", "eval")  # type: CodeType

        # Look for the requested variables
        global_dict = self.available_symbols()
        self.__locals = {}

        # Modified variable names after passing in the ast visitor
        self.__expr_vars = ast_visitor._expr_vars
        self.__const_vars = frozenset({key.replace(".", "_"): value for key, value in context.properties.items()})

        # Original variables names from the expression
        self.__all_vars = frozenset(ast_visitor._vars)
        self.__unconst_vars = frozenset(set(self.__all_vars) - set(context.properties))

        if isinstance(expression, Enum):
            etype = type(expression)
            global_dict = global_dict.copy()
            global_dict[etype.__name__] = etype

        self.__constant = False
        if set(self.__expr_vars).issubset(self.__const_vars):
            self.__constant = True
        else:
            required = set(self.__expr_vars)
            self.__constant = required and required.issubset(self.__const_vars)

        if self.__constant:
            value = eval(code, global_dict, self.locals)
            # simply return constant value
            eval_impl = self._EvalConstant(value)
        else:            
            # By specifying global and local contexts, we limit the user scope.
            eval_impl = self._EvalNotConstant(context, ast.unparse(ast_visited), code, self.__expr_vars)
        self._eval = eval_impl  # type: Callable[[], Any]

    def __getstate__(self) -> dict[str, Any]:
        """Creates a state of the object.

        The state type does NOT match type specified in
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        to allow custom serialization.

        Returns
        -------
        dict[str, Any]:
            state
        """
        return object__getstate__(self)

    def __json__(self) -> dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        dict[str, Any]
            The dictionary
        """
        return {"expression": self.__str}

    @staticmethod
    def string(expression: Any) -> str:
        """Converts an expression into a suitably formatted string.

        Notes
        -----
        Necessary step for numpy arrays (and possibly other types), whose plain string
        conversion is not readily evaluable (hence the use of 'repr' instead of 'str').

        Examples
        --------
        >>> str(numpy.array([0.1, 0.2]))
        [0.1 0.2]
        
        >>> repr(numpy.array([0.1, 0.2]))
        array([0.1, 0.2])
        """
        if isinstance(expression, str):
            # Include a substitution pass to make sure that no spaces are left
            # between objects and attributes, that is "foo.bar" instead of "foo .  bar"
            return re.sub("(?![0-9]) *\. *(?![0-9])", ".", expression.strip())
        elif isinstance(expression, numpy.ndarray):
            return repr(expression)
        else:
            return str(expression)

    def __str__(self) -> str:
        return self.__str

    def __repr__(self) -> str:
        return repr(self.__str)

    def __contains__(self, pattern: str):
        return (pattern in self.__str)

    @property
    def locals(self) -> dict[str, Any]:
        """dict[str, Any]: Context attributes required to evaluate the string expression."""
        # Read attribute values from context or from variable reference mapping of the context
        for key, (port, name) in self.__expr_vars.items():
            self.__locals[key] = getattr(port, name)
        return self.__locals

    @property
    def globals(self) -> dict[str, Any]:
        """dict[str, Any]: Global functions and variables required to evaluate the string expression."""
        return EvalString.__globals

    @property
    def eval_context(self) -> System:
        """cosapp.systems.System: Context of string expression evaluation."""
        return self.__context

    @property
    def constant(self) -> bool:
        """bool: `True` if evaluated expression is constant,
        that is independent of its context; `False` otherwise.
        """
        return self.__constant

    def eval(self) -> Any:
        """Evaluate the expression in the system context.

        Returns
        -------
        Any
            The result of the expression evaluation.
        """
        return self._eval()

    @property
    def variables(self) -> frozenset[str]:
        """frozenset[str]: Variables without system constant properties required for the evaluation of the expression."""
        return self.__unconst_vars

    @property
    def all_variables(self) -> frozenset[str]:
        """frozenset[str]:  All variables required for the evaluation of the expression."""
        return self.__all_vars

    @property
    def constants(self) -> frozenset[str]:
        """frozenset[str]: System constant properties required for the evaluation of the expression."""
        return self.__all_vars - self.__unconst_vars

    def __eq__(self, other: EvalString) -> bool:
        try:
            return self.__context is other.__context and self.__str == other.__str
        except:
            return False


class AssignString:
    """Create an executable assignment of the kind 'lhs = rhs' from two evaluable expressions lhs and rhs.
    """
    def __init__(self, lhs: str, rhs: Any, context: System) -> None:
        elhs = EvalString(lhs, context)
        if elhs.constant:
            raise ValueError(
                f"The left-hand side of an assignment expression cannot be constant ({elhs!r})")
        # At this point, lhs is a valid expression within given context
        self.__sides = None
        self.__raw_sides = [str(elhs), str(rhs)]  # raw sides lhs and rhs, without reformatting
        value = elhs.eval()
        if isinstance(value, numpy.ndarray):
            self.__shape = value.shape
            self.__dtype = value.dtype
        else:
            self.__shape = None
            self.__dtype = type(value)
        self.__context = context
        self.__lhs_vars = elhs.variables
        self.__rhs_vars = frozenset()
        self.__locals = elhs.locals.copy()
        self.__locals.update({"rhs_value": value, context.name: context})
        self._assignment = f"{context.name}.{elhs!s} = rhs_value"
        self.__code = compile(self._assignment, "<string>", "single")  # assignment bytecode
        self.rhs = rhs

    def __getstate__(self) -> dict[str, Any]:
        """Creates a state of the object.
        
        The state type depend on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.
        
        Returns
        -------
        dict[str, Any]:
            state
        """
        
        state = object__getstate__(self).copy()
        state.pop("_AssignString__code")
        return state
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        """Sets the object from a provided state.

        Parameters
        ----------
        state : dict[str, Any]
            State
        """
        self.__dict__.update(state)
        self.__code = compile(state["_assignment"], "<string>", "single")

    def __json__(self) -> dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        dict[str, Any]
            The dictionary
        """
        state = self.__getstate__()
        state.pop("_AssignString__context")
        if self.__context.name in state["_AssignString__locals"]:
            state["_AssignString__locals"].pop(self.__context.name)
        return state

    @property
    def eval_context(self) -> System:
        """cosapp.systems.System: Evaluation context of the assignment."""
        return self.__context

    @property
    def lhs(self) -> str:
        """str: Left-hand side of the assignment."""
        return self.__raw_sides[0]

    @property
    def lhs_variables(self) -> frozenset[str]:
        """frozenset[str]: set of variable names in left-hand side."""
        return self.__lhs_vars

    @property
    def rhs_variables(self) -> frozenset[str]:
        """frozenset[str]: set of variable names in right-hand side."""
        return self.__rhs_vars

    @property
    def contextual_lhs(self) -> str:
        """str: Contextual name of assignment left-hand side."""
        return f"{self.__context.name}.{self.lhs}"

    @property
    def rhs(self) -> str:
        """str: Right-hand side of the assignment."""
        return self.__raw_sides[1]

    @rhs.setter
    def rhs(self, rhs: Any) -> None:
        context = self.eval_context
        lhs = self.lhs
        erhs = EvalString(rhs, context)
        raw_rhs = str(erhs)
        value = erhs.eval()
        if erhs.constant:
            rhs = EvalString.string(value)
        else:
            rhs = raw_rhs
        sides = EvalString(f"({lhs}, {rhs})", context)
        if self.__shape:
            if value is None:
                value = sides.eval()[1]
            if isinstance(value, Number):
                sides = EvalString(f"({lhs}, full_like({lhs}, {rhs}))", context)
            else:  # tentatively copy rhs into a numpy array
                self.__check_size(value)
                sides = EvalString(f"({lhs}, array({rhs}, dtype={self.__dtype}))", context)
        self.__sides = sides
        self.__constant = erhs.constant
        self.__rhs_vars = erhs.variables
        self.__raw_sides[1] = raw_rhs

    @property
    def constant(self) -> bool:
        """bool: `True` if assignment right-hand side is constant, `False` otherwise."""
        return self.__constant

    def exec(self, context=None) -> tuple[Any, bool]:
        """
        Evaluates rhs, and executes assignment lhs <- rhs.

        Returns
        -------
        tuple[Any, bool]
            (rhs, changed), where 'changed' is True if the value of rhs has changed, False otherwise.
        """
        sides = self.__sides.eval()  # updates context at the same time
        changed = not numpy.array_equal(sides[0], sides[1])
        if context is None:
            context = self.__locals
        context['rhs_value'] = sides[1]
        exec(self.__code, dict(), context)
        return sides[1], changed

    @property
    def shape(self) -> tuple[int, int] | None:
        """tuple[int, int] | None: shape of assigned object (lhs) if it is an array, else None."""
        return self.__shape

    def variables(self) -> frozenset[str]:
        """Extracts all variables required for the assignment
        
        Returns
        -------
        frozenset[str]:
            Variable names as a set of strings
        """
        return self.__lhs_vars.union(self.__rhs_vars)

    def __str__(self) -> str:
        return " = ".join(self.__raw_sides)

    def __repr__(self) -> str:
        cls_name = self.__class__.__qualname__
        return "{}({!r}, {!r}, {})".format(cls_name, *self.__raw_sides, self.__context.name)

    def __check_size(self, array) -> None:
        """Checks if `array` is shape-compatible with lhs."""
        shape = self.__shape
        if numpy.shape(array) != shape:
            raise ValueError(f"Cannot assign {array} to array of shape {shape}")
