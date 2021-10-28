import sys
from pathlib import Path

import pytest

import cosapp.tests as test
from cosapp.utils.distributions import Uniform
from cosapp.systems import System
from cosapp.tests.library.ports import XPort
from cosapp.tests.library.systems.vectors import Strait1dLine


@pytest.fixture
def test_library():
    library_path = Path(test.__file__).parent / "library" / "systems"

    # Add path to allow System to find the component
    sys.path.append(str(library_path))
    try:
        yield library_path
    finally:
        # Undo path modification
        sys.path.remove(str(library_path))


@pytest.fixture
def test_data():
    return Path(test.__file__).parent / "data"


@pytest.fixture
def set_master_system():
    """Ensure the System class variable master is properly restored"""
    System._System__master_set = True
    try:
        yield
    finally:
        System._System__master_set = False


class Multiply2(System):
    def setup(self):
        self.add_input(XPort, "p_in", {"x": 1.0})
        self.add_inward("K1", 5.0, distribution=Uniform(worst=-0.1, best=+0.1))
        self.add_inward("K2", 5.0, distribution=Uniform(worst=-0.2, best=+0.2))
        self.add_output(XPort, "p_out", {"x": 1.0})

        self.add_outward("Ksum", 0.0)

    def compute(self):
        self.p_out.x = self.p_in.x * self.K1 * self.K2
        self.Ksum = self.K1 + self.K2


@pytest.fixture(scope="function")
def DummyFactory():
    class DummyMultiply(Multiply2):
        def setup(self, unknown=None, equation=None):
            super().setup()
            if unknown is not None:
                self.add_unknown(unknown)
            if equation is not None:
                self.add_equation(equation)

    return DummyMultiply


@pytest.fixture(scope="function")
def hat():
    s = System("hat")
    one = s.add_child(Strait1dLine("one"), pulling="in_")
    two = s.add_child(Strait1dLine("two"), pulling="out")
    s.connect(two.in_, one.out)
    return s


@pytest.fixture(scope="function")
def hat_case(hat):
    def factory(CaseDriver):
        case = CaseDriver("case")
        hat.add_driver(case)
        return hat, case

    return factory
