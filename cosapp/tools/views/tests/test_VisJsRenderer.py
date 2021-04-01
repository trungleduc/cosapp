import pytest

from cosapp.tests.library.systems import FanComplex
from cosapp.tools.views.visjs import VisJsRenderer
from cosapp.utils.testing import no_exception


pytest.importorskip("jinja2")


@pytest.fixture(scope='function')
def renderer():
    fan = FanComplex('fan')
    return VisJsRenderer(fan)


def test_VisJsRenderer_html_content(renderer):
    with no_exception():
        renderer.html_content()
