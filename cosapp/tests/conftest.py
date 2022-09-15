import pytest
import sys
from pathlib import Path
from typing import Type

from cosapp.systems import System


@pytest.fixture
def test_library():
    library_path = Path(__file__).parent / "library" / "systems"

    # Add path to allow System to find the component
    sys.path.append(str(library_path))
    try:
        yield library_path
    finally:
        # Undo path modification
        sys.path.remove(str(library_path))


@pytest.fixture
def test_data():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def DummySystemFactory():
    """Factory creating a dummy system class with custom attributes"""
    # mapping option / method
    # for example: `inputs` <-> `add_input`
    basics = (
        "inputs", "outputs",
        "inwards", "outwards",
        "transients", "rates",
        "properties",
    )
    extra = (
        "unknowns", "equations",
        "targets", "design_methods",
    )
    mapping = dict(
        (option, f"add_{option[:-1]}")
        for option in basics + extra
    )
    mapping["properties"] = "add_property"

    def Factory(classname, **settings):
        struct_methods = dict(
            (mapping[option], settings.get(option, None))
            for option in basics
        )
        extra_methods = dict(
            (mapping[option], settings.get(option, None))
            for option in extra
        )
        base: Type[System] = settings.get("base", System)

        class Prototype(base):
            def setup(self, **kwargs):
                super().setup(**kwargs)
                def add_attributes(method_dict: dict):
                    for method, values in method_dict.items():
                        if values is None:
                            continue
                        if not isinstance(values, list):
                            values = [values]
                        for info in values:
                            try:
                                args, kwargs = info  # expects a list of (tuple, dict)
                            except ValueError:
                                args, kwargs = [info], {}  # fallback
                            getattr(self, method)(*args, **kwargs)
                # Add inputs, outputs, transients, etc.
                add_attributes(struct_methods)
                # Add unknowns, equations & design methods
                add_attributes(extra_methods)
        
        return type(classname, (Prototype,), {})

    return Factory
