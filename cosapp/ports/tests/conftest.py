import pytest

from cosapp.base import Port
from typing import Any


Args, Kwargs = tuple[Any], dict[str, Any]


@pytest.fixture(scope="function")
def PortClassFactory():
    """Factory creating a new port class with custom attributes
    """
    def Factory(classname: str, variables: list[tuple[Args, Kwargs]], base: type[Port]=Port) -> type[Port]:
        # variables should be (args, kwargs) or a list thereof
        method_dict = {
            # correspondance method / option
            "add_variable": variables,
        }
        class PrototypePort(base):
            def setup(self, **options):
                super().setup(**options)
                for method, values in method_dict.items():
                    if values is None:
                        continue
                    if not isinstance(values, list):
                        values = [values]
                    for args, kwargs in values:  # expects a list of (tuple, dict)
                        getattr(self, method)(*args, **kwargs)
        return type(classname, (PrototypePort,), {})
    return Factory


@pytest.fixture(scope="function")
def PortFactory(PortClassFactory):
    """Factory creating a dummy port with custom attributes
    """
    def Factory(name: str, direction, init_values=None, owner=None, **options):
        PortClass: type[Port] = PortClassFactory(
            classname="PrototypePort",
            variables=options.pop('variables', []),
            base=options.pop('base', Port),
        )
        port = PortClass(name, direction, init_values, **options)
        if owner is not None:
            port.owner = owner
        return port
    return Factory
