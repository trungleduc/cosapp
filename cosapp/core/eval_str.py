"""
Module handling the execution of code provided as string by the user.

This code is inspired from the OpenMDAO module openmdao.components.exec_comp.
"""
import abc
from numbers import Number
from typing import Any, Dict, Iterable, NoReturn, Optional, Set, Tuple, Union

import numpy

from cosapp.utils.helpers import check_arg


class ContextLocals(dict):
    """Sub-set of context attributes.
    
    Parameters
    ----------
    context: System
        System whose attributes are looked up
    """

    def __init__(self, context: "System", *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.__context = context
    
    @property
    def context(self) -> "System":
        """cosapp.systems.System: Context of the locals"""
        return self.__context

    def __missing__(self, key: Any) -> Any:
        try:
            value = getattr(self.__context, key)
        except AttributeError:
            raise KeyError(key)
        else:
            self[key] = value
            return value


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

    Parameters
    ----------
    expression : str
        An Python statement. In addition to standard Python operators, a subset of numpy and scipy
        functions is supported.
    context : cosapp.core.module.Module or cosapp.core.numerics.basics.Residue
        System or Residue defining the local context in which the statement will be executed

    Notes
    -----
    The context is used to included the `System` ports, children, inwards and outwards.
    If the context is a `Residue`, the reference value from the residue is added as `residue_reference` variable in the
    execution context.
    """

    # this dict will act as the global scope when we eval our expressions
    __globals = {}  # type: Dict[str, Any]

    @classmethod
    def list_available_function(cls) -> Dict[str, Any]:
        """List the available functions in the execution context.

        Returns
        -------
        Dict[str, Any]
            Mapping of the available functions by their name.
        """
        if len(cls.__globals) > 0:
            return cls.__globals

        def _import_functs(
            mod: object, dct: Dict[str, Any], names: Optional[Iterable[str]] = None
        ) -> NoReturn:
            """
            Map attributes names from the given module into the given dict.

            Parameters
            ----------
            mod : object
                Module to check.
            dct : dict
                Dictionary that will contain the mapping
            names : iter of str, optional
                If supplied, only map attrs that match the given names
            """
            if names is None:
                names = dir(mod)
            for name in names:
                if isinstance(name, tuple):
                    name, alias = name
                else:
                    alias = name
                if not name.startswith("_"):
                    dct[name] = getattr(mod, name)
                    dct[alias] = dct[name]

        _import_functs(
            numpy,
            cls.__globals,
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
                # Array functions
                "array",
                "asarray",
                "arange",
                "concatenate",
                "ones",
                "zeros",
                "full",
                "full_like",
                "linspace",  # Array creation
                "e",
                "pi",  # Constants
                "inf",
                "isinf",
                "isnan",  # Logic
                "log",
                "log10",
                "log1p",
                "power",
                "sqrt",  # Math operations
                "cbrt",
                "exp",
                "expm1",
                "fmax",
                "fmin",
                "maximum",
                "minimum",
                "round",
                "sum",
                "dot",
                "prod",  # Reductions
                "tensordot",
                "matmul",  # Linear algebra
                "cross",
                "outer",
                "inner",
                "kron",
                "sin",
                "cos",
                "tan",
                ("arcsin", "asin"),  # Trig
                ("arccos", "acos"),
                ("arctan", "atan"),
                "sinh",
                "cosh",
                "tanh",
                ("arcsinh", "asinh"),  # Hyperbolic trig
                ("arccosh", "acosh"),
            ],
        )
        _import_functs(
            numpy.linalg, cls.__globals, names=["norm"]
        )

        # if scipy is available, add some functions
        try:
            import scipy.special
        except ImportError:
            pass
        else:
            _import_functs(
                scipy.special, cls.__globals, names=["factorial", "erf", "erfc"]
            )

        # Put any functions here that need special versions to work under
        # complex step

        def _cs_abs(x):
            if isinstance(x, numpy.ndarray):
                return x * numpy.sign(x)
            elif x.real < 0.0:
                return -x
            return x

        cls.__globals["abs"] = _cs_abs

        # Add residues helper function
        from cosapp.core.numerics.residues import Residue
        cls.__globals["evaluate_residue"] = Residue.evaluate_residue
        cls.__globals["residue_norm"] = Residue.residue_norm

        return cls.__globals

    def __init__(self, expression: Any, context: "System") -> NoReturn:
        """Class constructor.

        Compiles an expression, and checks that it is evaluable within a given context.
        """
        from cosapp.systems import System
        if not isinstance(context, System):
            raise TypeError(
                f"Object '{type(context)}' is not a valid context to evaluate expression '{expression}'."
            )

        self.list_available_function()

        self.__str = EvalString.string(expression)  # type: str

        code = compile(self.__str, "<string>", "eval")  # type: CodeType

        # Look for the requested variables
        self.__locals = ContextLocals(context)  # type: ContextLocals
        value = eval(code, EvalString.__globals, self.__locals)

        self.__constant = len(self.__locals) == 0  # type: bool

        if self.__constant:
            # simply return constant value
            self.__eval = lambda *args, **kwargs: value  # type: Callable[[], Any]
        else:            
            def eval_impl():
                # By specifying global and local contexts, we limit the user power.
                return eval(code, self.globals, self.locals)
            self.__eval = eval_impl  # type: Callable[[], Any]

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
            return expression.strip()
        elif isinstance(expression, EvalString):
            return str(expression)
        else:
            return repr(expression)

    def __str__(self) -> str:
        return self.__str

    def __repr__(self) -> str:
        return repr(self.__str)

    def __contains__(self, pattern: str):
        return (pattern in self.__str)

    @property
    def locals(self) -> Dict[str, Any]:
        """Dict[str, Any]: Context attributes required to evaluate the string expression."""
        # Read attribute values from context object
        eval_context = self.eval_context
        for key in self.__locals:
            self.__locals[key] = getattr(eval_context, key)
        return self.__locals

    @property
    def globals(self) -> Dict[str, Any]:
        """Dict[str, Any]: Global functions and variables required to evaluate the string expression."""
        return EvalString.__globals

    @property
    def eval_context(self) -> "System":
        """cosapp.systems.System: Context of string expression evaluation."""
        return self.__locals.context

    @property
    def constant(self) -> bool:
        """
        Returns a boolean indicating whether or not the evaluated expression
        is constant, that is independent of its context.
        """
        return self.__constant

    def eval(self) -> Any:
        """Evaluate the expression in the system context.

        Returns
        -------
        Any
            The result of the expression evaluation.
        """
        return self.__eval()


class AssignString:
    """Create an executable assignment of the kind 'lhs = rhs' from two evaluable expressions lhs and rhs."""

    def __init__(self, lhs: str, rhs: Any, context: "System") -> NoReturn:
        lhs = EvalString(lhs, context)
        if lhs.constant:
            raise ValueError(
                f"The left-hand side of an assignment expression cannot be constant ({lhs!r})")
        # At this point, lhs is a valid expression within given context
        self.__sides = None
        self.__raw_sides = [str(lhs), str(rhs)]  # raw sides lhs and rhs, without reformatting
        value = lhs.eval()
        if isinstance(value, numpy.ndarray):
            self.__shape = value.shape
            self.__dtype = value.dtype
        else:
            self.__shape = None
            self.__dtype = type(value)
        self.__context = context
        self.__locals = lhs.locals
        self.__locals.update({"rhs_value": value, context.name: context})
        assignment = f"{context.name}.{lhs!s} = rhs_value"
        self.__code = compile(assignment, "<string>", "exec")  # assignment bytecode
        self.rhs = rhs

    @property
    def eval_context(self) -> "System":
        """Evaluation context of the assignment."""
        return self.__context

    @property
    def lhs(self) -> str:
        """Left-hand side of the assignment, as a string"""
        return self.__raw_sides[0]

    @property
    def rhs(self) -> str:
        """Right-hand side of the assignment, as a string"""
        return self.__raw_sides[1]

    @rhs.setter
    def rhs(self, rhs: Any) -> NoReturn:
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
        self.__raw_sides[1] = raw_rhs

    @property
    def constant(self) -> bool:
        """Bool: is the right-hand side of the assignment constant?"""
        return self.__constant

    def exec(self, context=None) -> Tuple[Any, bool]:
        """
        Evaluates rhs, and executes assignment lhs <- rhs.

        Returns
        -------
        Tuple[Any, bool]
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
    def shape(self) -> Union[Tuple[int, int], None]:
        """Returns the shape of the assigned object (lhs) if it is an array, else None."""
        return self.__shape

    def __str__(self) -> str:
        return " = ".join(self.__raw_sides)

    def __repr__(self) -> str:
        cls_name = self.__class__.__qualname__
        return "{}({!r}, {!r}, {})".format(cls_name, *self.__raw_sides, self.__context.name)

    def __check_size(self, array):
        """Checks if `array` is shape-compatible with lhs."""
        shape = self.__shape
        if numpy.shape(array) != shape:
            raise ValueError(f"Cannot assign {array} to array of shape {shape}")
