"""Dummy call on D3 views"""
import pytest

from cosapp.tests.library.systems import FanComplex
from cosapp.tools.views.d3js import to_d3


pytest.importorskip("jinja2")


def test_to_d3():
    f = FanComplex('fan')
    to_d3(f, show=False)
