"""Dummy call on VIS js views"""
import os
import tempfile

import pytest

from cosapp.tests.library.systems import FanComplex
from cosapp.tools.views.visjs import to_visjs


pytest.importorskip("jinja2")


def test_to_visjs():
    f = FanComplex('fan')
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp:
        name = temp.name
    to_visjs(f, name)
    os.unlink(name)
