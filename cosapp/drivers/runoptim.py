from numbers import Number
from typing import Iterable, List, Tuple, Union, Optional

import logging
import numpy

from cosapp.core.eval_str import EvalString
from cosapp.core.numerics.basics import MathematicalProblem
from cosapp.core.numerics.boundary import Unknown
from cosapp.drivers.iterativecase import IterativeCase
from cosapp.drivers.utils import UnknownAnalyzer
from cosapp.utils.helpers import check_arg

logger = logging.getLogger(__name__)


class RunOptim(IterativeCase):
    """Driver running the model on its `System` owner and gathering optimization definition.

    Parameters
    ----------
    name : str
        Name of the driver
    owner : System, optional
        :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
    **kwargs : Dict[str, Any]
        Optional keywords arguments
    """

    __slots__ = ('__problem', 'constraints',)

    def __init__(self,
        name: str,
        owner: "Optional[cosapp.systems.System]" = None,
        **kwargs
    ) -> None:
        """Initialize a driver

        Parameters
        ----------
        name: str, optional
            Name of the `Module`
        owner : System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
        **kwargs : Dict[str, Any]
            Optional keywords arguments
        """
        super().__init__(name, owner, **kwargs)
        # TODO we need to move this in an enhanced MathematicalProblem
        self.constraints = list()  # type: List[Dict]
            # desc="Constraints to be applied in the optimization problem"

    def set_objective(self, expression: str) -> None:
        """Set the scalar objective function to be minimized.

        Parameters
        ----------
        expression : str
            The objective expression to be minimized.
        """
        if self.owner is None:
            raise AttributeError("Owner System is needed to define an optimization.")

        check_arg(expression, "expression", str, lambda s: "==" not in s)

        self.__problem.residues.clear()  # Ensure only one objective is defined
        self.__problem.add_equation(f"{expression} == 0", name="f_objective", reference=1.)

    def add_unknown(self,
            name: Union[str, Iterable[Union[dict, str, Unknown]]],
            max_abs_step: Number = numpy.inf,
            max_rel_step: Number = numpy.inf,
            lower_bound: Number = -numpy.inf,
            upper_bound: Number = numpy.inf
    ) -> "MathematicalProblem":
        """Add unknown variables.

        You can set variable one by one or provide a list of dictionary to set multiple variable at once. The
        dictionary key are the arguments of this method.

        Parameters
        ----------
        name : str or Iterable of dictionary or str
            Name of the variable or list of variable to add
        max_rel_step : float, optional
            Maximal relative step by which the variable can be modified by the numerical solver; default numpy.inf
        max_abs_step : float, optional
            Maximal absolute step by which the variable can be modified by the numerical solver; default numpy.inf
        lower_bound : float, optional
            Lower bound on which the solver solution is saturated; default -numpy.inf
        upper_bound : float, optional
            Upper bound on which the solver solution is saturated; default numpy.inf

        Returns
        -------
        MathematicalProblem
            The modified MathematicalSystem
        """
        self.__problem.add_unknown(name, max_abs_step, max_rel_step, lower_bound, upper_bound)

    def add_constraints(self,
        expression: Union[str, List[Union[str, Tuple[str, bool]]]],
        inequality: bool = True,
    ) -> None:
        """Add constraints to the optimization problem.

        Parameters
        ----------
        expression : str
            The expression defining the constraint
        inequality : bool
            If True, expression must be non-negative; else must be zero.
        """
        def add_constraint(expression: str, inequality: bool) -> None:
            check_arg(expression, 'expression', str)
            check_arg(inequality, 'inequality', bool)

            # Test that the expression can be evaluated
            EvalString(expression, self.owner).eval()

            self.constraints.append({
                'type': 'ineq' if inequality else 'eq',
                'formula': expression
            })

        if self.owner is None:
            raise AttributeError("Owner System is required to define an optimization.")

        if isinstance(expression, str):
            add_constraint(expression, inequality)
        else:
            for args in expression:
                if isinstance(args, str):
                    add_constraint(args, inequality)
                else:
                    add_constraint(*args)

    def reset_problem(self) -> None:
        self.__problem = MathematicalProblem(self.name, self.owner)  # type: MathematicalProblem

    def get_problem(self) -> MathematicalProblem:
        """Returns the full mathematical for the case.

        Returns
        -------
        MathematicalProblem
            The full mathematical problem to solve for the case
        """
        # TODO Go further to gather optimization unknowns defined in System hierarchy
        return self.__problem

    def setup_run(self):
        """Method called once before starting any simulation."""
        super().setup_run()
        
        # Resolve unknown aliasing and connected unknowns
        analyzer = UnknownAnalyzer(self.owner)
        self.__problem = analyzer.filter_problem(self.__problem)
