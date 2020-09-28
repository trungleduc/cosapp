import numpy as np
import pytest

from cosapp.systems import System
from cosapp.core.numerics.residues import AbstractResidue


class ASyst(System):
    def setup(self):
        self.add_inward("a", 1.0)
        self.add_inward("b", [1.0, 2.0])
        self.add_inward("c", np.asarray([1.0, 2.0]))


class DummyResidue(AbstractResidue):
    def update():
        pass

    def copy():
        pass


def test_AbstractResidue___init___():
    s = ASyst("a_system")
    d = DummyResidue(s, "a_residue")

    for a, v in ({"reference": 1.0, "name": "a_residue"}).items():
        assert getattr(d, a) == v

    assert d._context is s
    assert d.value is None


def test_AbstractResidue__get_zeros_rhs():
    s = ASyst("a_system")
    d = DummyResidue(s, "a_residue")

    assert d._get_zeros_rhs(1.0) == 0.0
    assert np.array_equal(d._get_zeros_rhs([1.0, 2.0]), np.zeros(2))
    assert np.array_equal(d._get_zeros_rhs(np.asarray([1.0, 2.0])), np.zeros(2))

    assert d._get_zeros_rhs("1.") == "zeros(())"
    assert d._get_zeros_rhs("[1., 2.]") == "zeros((2,))"
    assert d._get_zeros_rhs("array([1., 2.])") == "zeros((2,))"


def test_AbstractResidue_reference_setter():
    s = ASyst("a_system")
    d = DummyResidue(s, "a_residue")

    assert d.reference == 1.0
    d.reference = 22.0
    assert d.reference == 22.0


def test_AbstractResidue_context():
    s = ASyst("a_system")
    d = DummyResidue(s, "a_residue")

    assert d.context is s
    with pytest.raises(AttributeError):
        d.context = System("foo")


def test_AbstractResidue_name():
    s = ASyst("a_system")
    d = DummyResidue(s, "a_residue")

    assert d.name == "a_residue"
    with pytest.raises(AttributeError):
        d.name = "banana"


def test_AbstractResidue_value():
    s = ASyst("a_system")
    d = DummyResidue(s, "a_residue")

    assert d.value is None
    with pytest.raises(AttributeError):
        d.value = "banana"
