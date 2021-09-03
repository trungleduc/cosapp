import numpy, scipy.interpolate
import enum
import logging
from typing import Any, Dict, List

from cosapp.drivers.driver import Driver
from cosapp.core.numerics.boundary import Boundary
from cosapp.core.eval_str import AssignString
from cosapp.utils.helpers import check_arg

logger = logging.getLogger(__name__)


class Interpolator:
    """Class describing a function defined from tabulated data,
    based on interpolating function `interpolator`. Function `interpolator`
    must return a callable function x -> f(x), interpolating [y_0, ..., y_n] at
    points [x_0, ..., x_n]."""

    class Kind(enum.Enum):
        """Interpolator kind"""
        Linear = scipy.interpolate.interp1d
        CubicSpline = scipy.interpolate.CubicSpline
        Pchip = scipy.interpolate.PchipInterpolator
    
    def __init__(self, data, kind=Kind.Linear):
        self.__data = None
        self.kind = kind
        self.data = data

    @property
    def kind(self) -> Kind:
        return self.__kind

    @kind.setter
    def kind(self, kind: Kind) -> None:
        check_arg(kind, 'kind', self.Kind)
        self.__kind = kind
        self.__update_evaluator()

    @property
    def data(self) -> numpy.ndarray:
        return self.__data
    
    @data.setter
    def data(self, data) -> None:
        data = numpy.asarray(data, dtype=float)
        shape = data.shape
        if data.ndim != 2:
            raise ValueError(f"invalid shape {shape}; `data` must be a 2D array.")
        if shape[1] == 2:
            # data given as [[x0, y0], ..., [xn, yn]]
            data = data.T
        # sort data to ensure increasing x
        self.__data = data[:, data[0].argsort()]
        self.__data.setflags(write=0)
        self.__update_evaluator()

    def __update_evaluator(self) -> None:
        data = self.data
        if data is None:
            def evaluator(x):
                raise ValueError("data was not specified")
        else:
            evaluator = self.kind.value(data[0], data[1])
        self.__evaluator = evaluator

    def __call__(self, t: float) -> numpy.ndarray:
        return self.__evaluator(t)


class InterpolAssignString:
    """
    Create an executable assignment to handle time boundary conditions of the kind `variable = F(t, data)`,
    where F is an interpolation function of some dataset at time t.
    This class is very similar to `cosapp.core.eval_str.AssignString`, except the right-hand side is a function.
    In order to limit the scope of the callable function, the rhs can only be of type `Interpolator`.
    """
    def __init__(self, variable, function, context):
        # Strict type check (as opposed to `isinstance`), to ensure the type of
        # `function` is `Interpolator`, and is not derived from `Interpolator`.
        ftype = type(function)
        if ftype is not Interpolator:
            raise TypeError(
                f"Functions used in time boundary conditions may only be of type `Interpolator`; got {ftype}"
            )
        Boundary.parse(context, variable)  # checks that variable is valid
        fname = f"BC{id(function)}"
        assignment = f"{context.name}.{variable} = float({fname}(t))"
        self.__context = context
        self.__locals = {fname: function, context.name: context, 't': 0}
        self.__code = compile(assignment, "<string>", "exec")  # type: CodeType
        self.__str = f"{variable} = {ftype.__name__}(t)"

    def exec(self) -> None:
        """
        Evaluates rhs, and executes assignment lhs <- rhs.
        """
        exec(self.__code, {}, self.locals)

    def __str__(self) -> str:
        return self.__str

    @property
    def locals(self) -> Dict[str, Any]:
        """Dict[str, Any]: Context attributes required to evaluate the string expression."""
        self.__locals['t'] = self.__context.t
        return self.__locals
    
    @property
    def constant(self):
        return False


class Scenario:
    """Class managing boundary and initial conditions for time simulations"""

    def __init__(self, name: str, owner: Driver) -> None:
        """Initialize object

        Parameters
        ----------
        name: str
            Name of the `Module`
        owner : Driver
            :py:class:`~cosapp.drivers.driver.Driver` to which object belong
        """
        self.__case_values = []   # type: List[AssignString]
        self.__init_values = []   # type: List[AssignString]
        self.name = name
        self.owner = owner

    @classmethod
    def make(cls, name: str, init: Dict[str, Any], values: Dict[str, Any], driver: Driver) -> "Scenario":
        """Scenario factory"""
        scenario = cls(name, driver)
        scenario.set_init(init)
        scenario.set_values(values)
        return scenario

    def apply_init_values(self) -> None:
        """Execute assignments corresponding to initial conditions"""
        logger.debug("Apply initial conditions")
        for assignment in self.__init_values:
            logger.debug(f"\t{assignment}")
            assignment.exec()

    def update_values(self) -> None:
        """Execute assignments corresponding to boundary conditions"""
        for assignment in self.__case_values:
            assignment.exec()

    @property
    def context(self) -> "System":
        """System: evaluation context of initial and boundary conditions,
        that is the system controled by owner driver"""
        return self.__context

    @property
    def owner(self) -> Driver:
        """Driver: owner driver"""
        return self.__owner

    @owner.setter
    def owner(self, driver: Driver) -> None:
        check_arg(driver, "owner", Driver, lambda driver: hasattr(driver, "owner"))
        self.__owner = driver
        self.__context = driver.owner
        self.clear_init()
        self.clear_values()

    def add_init(self, modifications: Dict[str, Any]) -> None:
        """Add a set of initial conditions, from a dictionary of the kind {'variable': value, ...}

        Parameters
        ----------
        modifications : Dict[str, Any]
            Dictionary of (variable name, value)

        Examples
        --------
        >>> scenario.add_init({'myvar': 42, 'port.dummy': '-2 * alpha'})
        """
        check_arg(modifications, 'modifications', dict)

        if self.owner is None:
            raise AttributeError(f"Driver {self.name!r} must be attached to a System to set initial values.")

        for variable, value in modifications.items():
            assignment = self.__assignment(variable, value)
            if assignment.constant:
                # If assignment is constant, it is safer to insert it at the top
                # of the list, since other initial condition assignments might use it.
                self.__init_values.insert(0, assignment)
            else:
                self.__init_values.append(assignment)

    def set_init(self, modifications: Dict[str, Any]) -> None:
        """Set initial conditions, from a dictionary of the kind {'variable': value, ...}

        See `add_init` for further detail.
        """
        self.clear_init()
        self.add_init(modifications)

    def add_values(self, modifications: Dict[str, Any]) -> None:
        """Add a set of variables to the list of case values, from a dictionary of the kind {'variable': value, ...}

        Each variable and its value can be contextual, as in {'child1.port2.var': '2 * child2.foo'},
        as long as they are both evaluable in the context of the driver's owner.
        Explicit time dependency can be given using variable 't' in values, as in 'exp(-t / tau)'

        Parameters
        ----------
        modifications : Dict[str, Any]
            Dictionary of (variable name, value)

        Examples
        --------
        >>> scenario.add_values({'myvar': 42, 'port.dummy': 'cos(omega * t)'})
        """
        check_arg(modifications, 'modifications', dict)

        if self.owner is None:
            raise AttributeError(f"Driver {self.name!r} must be attached to a System to set case values.")

        for variable, value in modifications.items():
            if type(value) is Interpolator:
                assignment = InterpolAssignString(variable, value, self.context)
            else:
                assignment = self.__assignment(variable, value)
            if assignment.constant:
                # If assignment is constant, it can be regarded as an initial condition,
                # rather than a time-dependent boundary condition.
                # Moreover, it is safer to insert it at the top of the list,
                # since other initial condition assignments might use it.
                self.__init_values.insert(0, assignment)
            else:
                self.__case_values.append(assignment)

    def set_values(self, modifications: Dict[str, Any]) -> None:
        """Set case values, from a dictionary of the kind {'variable': value, ...}

        See `add_values` for further detail.
        """
        self.clear_values()
        self.add_values(modifications)

    def clear_values(self) -> None:
        """Clears the list of boundary conditions"""
        self.__case_values.clear()

    def clear_init(self) -> None:
        """Clears the list of initial conditions"""
        self.__init_values.clear()

    @property
    def case_values(self) -> List[AssignString]:
        """List[AssignString]: list of boundary conditions"""
        return self.__case_values

    @property
    def init_values(self) -> List[AssignString]:
        """List[AssignString]: list of initial conditions"""
        return self.__init_values

    def __assignment(self, variable, value) -> AssignString:
        """Returns the AssignString corresponding to assignment `variable <- value`"""
        context = self.__context
        Boundary.parse(context, variable)  # checks that variable is valid
        return AssignString(variable, value, context)

    def __repr__(self) -> str:
        s = f"{type(self).__name__} {self.name!r}, in {self.context.name!r}"
        def conditions(assigments, title):
            return "\n  - ".join(
                [f"\n{title.title()}:"] + [str(a) for a in assigments]
             ) if assigments else ""
        s += conditions(self.init_values, "Initial values")
        s += conditions(self.case_values, "Boundary conditions")
        return s
