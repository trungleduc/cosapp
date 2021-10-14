from wrapt.wrappers import ObjectProxy
from cosapp.core.connectors import Connector


class SystemConnector(ObjectProxy):
    """Connector proxy used in `System`
    """
    def __init__(self, connector: Connector):
        if not isinstance(connector, Connector):
            raise TypeError(
                "SystemConnector can only wrap objects of type Connector"
            )
        super().__init__(connector)
        self.activate()

    @classmethod
    def make(cls, *args, **kwargs) -> "SystemConnector":
        """Factory returning a new connector proxy from
        `Connector` constructor arguments.

        Parameters:
        -----------
        *args, **kwargs [Any]:
            Arguments forwarded to `Connector` constructor.

        Returns:
        --------
        connector [SystemConnector]:
            Newly created `SystemConnector` proxy.
        """
        return cls(Connector(*args, **kwargs))

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

    def transfer(self) -> None:
        """Transfer values from `source` to `sink`."""
        if self.__is_active:
            self.__wrapped__.transfer()
