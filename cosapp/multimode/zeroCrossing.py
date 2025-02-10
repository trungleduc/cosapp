from __future__ import annotations
import enum
from typing import Dict, NamedTuple, Any, Callable
from cosapp.utils.helpers import check_arg
from cosapp.utils.state_io import object__getstate__


@enum.unique
class EventDirection(enum.Enum):
    """Enum covering zero-crossing directions"""
    UP = {
        'desc': "Upward zero-crossing",
        'func': (lambda prev, curr: curr > prev and curr * prev <= 0),
    }
    DOWN = {
        'desc': "Downward zero-crossing",
        'func': (lambda prev, curr: curr < prev and curr * prev <= 0),
    }
    UPDOWN = {
        'desc': "Up- or downward zero-crossing",
        'func': (lambda prev, curr: curr != prev and prev * curr <= 0),
    }

    def zero_detected(self, prev, curr) -> bool:
        detector = self.value['func']
        return detector(prev, curr)

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        state = self.__getstate__().copy()
        state.pop("_value_")
        return state

    @classmethod
    def _new(cls, value: str) -> EventDirection:
        return cls[value]

    def __reduce_ex__(self, _: Any) -> tuple[Callable, tuple, dict]:
        """Defines how to serialize/deserialize the object.
        
        Parameters
        ----------
        _ : Any
            Protocol used

        Returns
        -------
        tuple[Callable, tuple, dict]
            A tuple of the reconstruction method, the arguments to pass to
            this method, and the state of the object
        """
        return self._new, (self.name, ), {}
    
    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        state = object__getstate__(self).copy()
        state.pop("_value_")
        return state

class ZeroCrossing(NamedTuple):
    expression: str
    direction: EventDirection

    @classmethod
    def up(cls, expression: str) -> ZeroCrossing:
        return cls(expression, EventDirection.UP)

    @classmethod
    def down(cls, expression: str) -> ZeroCrossing:
        return cls(expression, EventDirection.DOWN)

    @classmethod
    def updown(cls, expression: str) -> ZeroCrossing:
        return cls(expression, EventDirection.UPDOWN)

    @classmethod
    def from_comparison(cls, expression: str) -> ZeroCrossing:
        """Interpret an expression of the kind 'lhs <op> rhs'
        as a `ZeroCrossing` object, where <op> is one of
        comparison operators:
        - `<`, `<=`
        - `==`
        - `>`, `>=`
        """
        check_arg(expression, "expression", str)
        operators = cls.operators()
        not_in_sides = list(operators) + ["="]

        def side_ok(side: str) -> bool:
            return not any(nope in side for nope in not_in_sides) and side.strip()

        for operator, direction in operators.items():
            try:
                lhs, rhs = expression.split(operator, maxsplit=1)
            except:
                continue
            if side_ok(lhs) and side_ok(rhs):
                return cls(f"{lhs.strip()} - ({rhs.strip()})", direction)

        raise ValueError(
            f"Expected a comparison between two expressions involving one of {tuple(operators)}"
            f"; got {expression!r}."
        )

    @classmethod
    def operators(cls) -> Dict[str, EventDirection]:
        return {
            "<" : EventDirection.DOWN,
            "<=": EventDirection.DOWN,
            "==": EventDirection.UPDOWN,
            ">=": EventDirection.UP,
            ">" : EventDirection.UP,
        }
