import pytest
import logging
import re

from cosapp.ports.port import PortType, ModeVarPort
from cosapp.base import System


@pytest.fixture(scope='module')
def context():
    return System('context')


def test_ModeVarPort_add_mode_variable(caplog, context):
    """Check that specifying an init value for """
    p_in = ModeVarPort('p_in', direction=PortType.IN)
    p_out = ModeVarPort('p_out', direction=PortType.OUT)
    p_in.owner = p_out.owner = context

    with caplog.at_level(logging.WARNING):
        p_in.add_mode_variable('a1', init=0.0)
        p_out.add_mode_variable('b1', init=0.0)
        p_in.add_mode_variable('a2', 1.0, init=0.0)
        p_out.add_mode_variable('b2', 1.0, init=0.0)

    assert len(caplog.records) == 1
    assert re.match(
        "Initial value .* discarded for input mode variable 'a2'",
        caplog.records[0].message
    )
