import pytest

import numpy as np
import logging
from collections.abc import Iterable
from typing import Any
from cosapp.ports import Port
from cosapp.systems import System
from cosapp.drivers import RunOnce, ValidityCheck


class XPort(Port):
    def setup(self):
        self.add_variable("x",
            valid_range=(0, 2),
            invalid_comment="Unreasonable set of value",
            limits=(-2, 4),
            out_of_limits_comment="Don't push your luck",
        )


class Simple(System):
    def setup(self, k_range=None, k_limits=None):
        self.add_input(XPort, "x_in")
        self.add_output(XPort, "x_out")
        self.add_inward("k",
            value=0.0,
            valid_range=k_range,
            invalid_comment="Unreasonable k",
            limits=k_limits,
            out_of_limits_comment="Forbidden k",
        )
        self.add_outward("y",
            valid_range=(0.5, 3.5),
            invalid_comment="Unreasonable y",
            limits=(0, None),
            out_of_limits_comment="Forbidden y",
        )

    def compute(self):
        val = sum(self.k) if isinstance(self.k, Iterable) else self.k
        self.x_out.x = self.x_in.x + val
        self.y = self.x_in.x * val


@pytest.fixture(scope="function")
def make_simple():
    def factory(settings=dict()):
        s = Simple('simple',
            k_range = settings.get('valid_range', None),
            k_limits = settings.get('limits', None),
        )
        s.add_driver(RunOnce("run"))
        s.add_driver(ValidityCheck("check"))
        s.k = settings.get('value', 0.0)
        return s

    return factory


def get_log(log: Any, lvl: int) -> str:
    ret = ""
    for record in log.record_tuples:
        if record[1] == lvl:
            ret = record[2]
    return ret


@pytest.mark.parametrize("settings, expected", [
    (dict(value=1), dict()),
    (
        dict(value=1.5, valid_range=(None, 2), limits=(None, 3)),
        dict(warning="\nx_out.x = 2.5 not in [0, 2] - Unreasonable set of value"),
    ),
    (
        dict(value=5, valid_range=(-2, 2), limits=(None, 3)),
        dict(
            warning = "\noutwards.y = 5 not in [0.5, 3.5] - Unreasonable y",
            error = "\nx_out.x = 6 not in [-2, 4] - Don't push your luck\n\tinwards.k = 5 not in [-inf, 3] - Forbidden k",
        ),
    ),
    (
        dict(value=-0.5, valid_range=(-2, 2), limits=(None, 3)),
        dict(error="\noutwards.y = -0.5 not in [0.0, inf] - Forbidden y"),
    ),
    (
        dict(value=-2, valid_range=(-2, 2), limits=(None, 3)),
        dict(
            warning = "\nx_out.x = -1 not in [0, 2] - Unreasonable set of value",
            error = "\noutwards.y = -2 not in [0.0, inf] - Forbidden y",
        ),
    ),
    (
        dict(value=np.array([3.3, 3]), valid_range=(-2, 2), limits=(None, 3)),
        dict(
            warning = "\noutwards.y = 6.3 not in [0.5, 3.5] - Unreasonable y",
            error = "\nx_out.x = 7.3 not in [-2, 4] - Don't push your luck\n\tinwards.k = [3.3 3. ] not in [-inf, 3] - Forbidden k",
        ),
    ),
    (
        dict(value=np.array([3.3, 3]), valid_range=(None, None), limits=(None, 3)),
        dict(
            warning = "\noutwards.y = 6.3 not in [0.5, 3.5] - Unreasonable y",
            error = "\nx_out.x = 7.3 not in [-2, 4] - Don't push your luck",
        ),
    ),
    (
        dict(value=np.array([3.3, 3]), limits=(None, 3)),
        dict(
            warning = "\noutwards.y = 6.3 not in [0.5, 3.5] - Unreasonable y",
            error = "\nx_out.x = 7.3 not in [-2, 4] - Don't push your luck\n\tinwards.k = [3.3 3. ] not in [-inf, 3] - Forbidden k"
        ),
    ),
    (
        dict(value=np.array([3.3, 3])),
        dict(
            warning = "\noutwards.y = 6.3 not in [0.5, 3.5] - Unreasonable y",
            error = "\nx_out.x = 7.3 not in [-2, 4] - Don't push your luck",
        ),
    ),
    (
        dict(value=np.array([3.3, 3, 2]), valid_range=((-2, 1), (2, 2.5), (0, 4)), limits=((-5, 5), (-5, 5), (-5, 6))),
        dict(
            warning = "\ninwards.k = [3.3 3.  2. ] not in [(-2, 1), (2, 2.5), (0, 4)] - Unreasonable k\n\toutwards.y = 8.3 not in [0.5, 3.5] - Unreasonable y",
            error = "\nx_out.x = 9.3 not in [-2, 4] - Don't push your luck",
        )
    ),
])
def test_ArrayValidityCheck(caplog, make_simple, settings, expected):
    s = make_simple(settings)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        s.run_drivers()
        
        assert get_log(caplog, logging.WARNING) == expected.get('warning', '')
        assert get_log(caplog, logging.ERROR) == expected.get('error', '')


@pytest.mark.parametrize("settings, expected", [
    (
        dict(value=1, valid_range=((2, 3), (2, 6)), limits=((2, 3), (2, 6))),
        dict(
            error = ValueError,
            match = r"valid_range \(.*\) or limits \(.*\) of variable 'simple.inwards.k' are incompatible with its value 1",
        ),
    ),
    (
        dict(value=np.array([3.3, 3]), valid_range=2, limits=(None, 3)),
        dict(
            error = TypeError,
            match = "Validity or limit range must be a tuple with format comparable to value",
        ),
    ),
    (
        dict(value=np.array([3.3, 3]), valid_range=(1, 2, 3), limits=(None, 3)),
        dict(
            error = TypeError,
            match = "Valid range or limits must be a size 2 tuple with type comparable to value",
        ),
    ),
    (
        dict(value=np.array([3.3, 3]), valid_range=((1, 2), (2, 3)), limits=(None, 3)),
        dict(
            error = ValueError,
            match = r"valid_range \(.*\) and limits \(None, 3\) of variable 'simple.inwards.k' have different formats",
        ),
    ),
    (
        dict(value=np.array([3.3, 3]), valid_range=((1, 2), (2, 3)), limits=((1, 5), 3)),
        dict(
            error = ValueError,
            match = r"Mixed values in valid_range \(\(1, 5\), 3\) of 'simple.inwards.k'",
        ),
    ),
])
def test_ArrayValidityException(make_simple, settings, expected):
    error = expected['error']

    with pytest.raises(error, match=expected.get('match', None)):
        s = make_simple(settings)
        s.run_drivers()


def test_ValidityCheck_port_property(caplog):
    """Validity check involving a port with a read-only property.
    Related to https://gitlab.com/cosapp/cosapp/-/issues/177
    """
    class PortWithProperty(Port):
        """Geometry port, with a property depending on port variable"""
        def setup(self) -> None:
            self.add_variable(
                name="height",
                unit="m",
                limits=(0.0, np.inf),
            )
            self.add_variable(
                name="width",
                unit="m",
                limits=(0.0, np.inf),
            )

        @property
        def area(self) -> float:
            return self.height * self.width

    class Model(System):
        def setup(self):
            self.add_input(PortWithProperty, "section")
            self.add_outward("section_area", 0.0, valid_range=(0.0, 0.1))

        def compute(self):
            self.section_area = self.section.area

    model = Model("model")
    model.add_driver(RunOnce("run"))
    model.add_driver(ValidityCheck("check"))

    model.section.height = 1.0
    model.section.width = 0.25

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model.run_drivers()

    assert len(caplog.records) == 1
    assert model.section_area == pytest.approx(0.25, rel=1e-14)
    assert "section_area = 0.25 not in [0.0, 0.1]" in caplog.messages[0]
