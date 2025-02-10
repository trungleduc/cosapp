import inspect
from cosapp.ports.port import BasePort
from cosapp.ports.connectors import BaseConnector
from cosapp.patterns import Proxy
from typing import Dict, Any, Callable


class SystemConnector(Proxy):
    """Connector proxy used in `System`"""

    def __init__(self, connector: BaseConnector):
        self.check(connector)
        super().__init__(connector)
        self.__noise: Dict[str, Any] = {}
        self.source.touch()
        self.activate()

    @staticmethod
    def check(wrappee: Any) -> None:
        """Checks whether `wrappee` can be wrapped in a `SystemConnector`
        proxy; raises an exception if not.

        Parameters:
        -----------
        - wrappee [Any]:
            If wrappee is a class, check that it is a concrete implementation of
            `BaseConnector`. If it is an object, check that its type is derived from
            `BaseConnector`.

        Raises:
        -------
        - `ValueError` if `wrappee` is a class not derived from `BaseConnector`
        - `TypeError` if `wrappee` is an object not derived from `BaseConnector`
        """
        if inspect.isclass(wrappee):
            ok = lambda t, base: issubclass(t, base) and t is not base
            error = ValueError
        else:
            ok = isinstance
            error = TypeError
        if not ok(wrappee, BaseConnector):
            raise error("`SystemConnector` can only wrap objects derived from `BaseConnector`")

    def __getstate__(self) -> Dict[str, Any]:
        """Creates a state of the object.

        The state type does NOT match type specified in
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        to allow custom serialization.

        Returns
        -------
        Dict[str, Any]:
            state
        """
        return {"__noise": self.__noise, "_wrapped": super().__getattribute__("_wrapped"), "__is_active": self.__is_active}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Sets the object from a provided state.

        Parameters
        ----------
        state : Dict[str, Any]
            State
        """
        self.__is_active = state["__is_active"]
        self._wrapped = state["_wrapped"]
        self.__noise = state["__noise"]

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        return {"_wrapped": self._wrapped}

    @property
    def is_active(self) -> bool:
        """bool: `True` if connector is activated, `False` otherwise."""
        return self.__is_active

    def activate(self) -> None:
        """Activate connector transfer."""
        self.__is_active = True

    def deactivate(self) -> None:
        """Deactivate connector transfer."""
        self.__is_active = False

    def set_perturbation(self, name: str, value: Any) -> None:
        """Add a perturbation on a connector.

        Parameters
        ----------
        name : str
            Name of the sink variable to perturb
        value : Any
            Perturbation value
        """
        if name not in self.sink_variables():
            raise ValueError("Perturbations can only be added on sink variables")
        self.__noise[name] = value
        self.sink.owner.touch()

    def clear_noise(self) -> None:
        self.__noise.clear()

    def __repr__(self):
        return repr(self._wrapped)

    def transfer(self) -> None:
        """Transfer values from `source` to `sink`."""
        if self.__is_active:
            sink: BasePort = self.sink
            source: BasePort = self.source
            noise = self.__noise

            if not source.is_clean or noise:
                sink.touch()
                self._wrapped.transfer()
                for varname, perturbation in noise.items():
                    sink[varname] += perturbation
