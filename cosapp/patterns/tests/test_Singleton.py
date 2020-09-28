import pytest

from cosapp.patterns import Singleton


class Unique(metaclass=Singleton):
    pass


class DerivedUnique(Unique):
    pass


def test_Unique_unicity():
    a = Unique()
    b = Unique()
    assert a is b


def test_DerivedUnique_unicity():
    a = DerivedUnique()
    b = DerivedUnique()
    assert a is b

    u = Unique()
    assert a is not u
