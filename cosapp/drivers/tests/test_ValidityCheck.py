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
        self.add_variable(
            "x",
            valid_range=(0, 2),
            invalid_comment="Reasonable set of value",
            limits=(-2, 4),
            out_of_limits_comment="Don't push your luck",
        )


@pytest.fixture(scope="function")
def Simple():
    def factory(name, k_val, k_valid_range, k_limits):
        class _Simple(System):
            def setup(self):
                self.add_input(XPort, "x_in")
                self.add_output(XPort, "x_out")
                self.add_inward(
                    "k",
                    value=k_val,
                    valid_range=k_valid_range,
                    invalid_comment="Reasonable k",
                    limits=k_limits,
                    out_of_limits_comment="Forbidden k",
                )
                self.add_outward(
                    "y",
                    valid_range=(0.5, 3.5),
                    invalid_comment="Reasonable y",
                    limits=(0, None),
                    out_of_limits_comment="Forbidden y",
                )

            def compute(self):
                val = sum(self.k) if isinstance(self.k, Iterable) else self.k

                self.x_out.x = self.x_in.x + val
                self.y = self.x_in.x * val
        return _Simple(name)
    return factory


def get_log(log: Any, lvl: int) -> str:
    ret = ""
    for record in log.record_tuples:
        if record[1] == lvl:
            ret = record[2]
    return ret


def test_ValidityCheck_setup(Simple):
    s = Simple("system", 2, None, None)
    s.add_driver(RunOnce("run"))
    d = s.add_driver(ValidityCheck("check"))

    assert d.name == "check"
    assert d.owner is s


@pytest.mark.parametrize("k_val, k_valid_range, k_limits, warning_msg, error_msg", [
    (1, None, None, "", ""),
    (1.5, (None, 2), (None, 3), "\nx_out.x = 2.5 not in [0, 2] - Reasonable set of value", ""),

    (5, (-2, 2), (None, 3),
        "\noutwards.y = 5 not in [0.5, 3.5] - Reasonable y",
        "\nx_out.x = 6 not in [-2, 4] - Don't push your luck\n\tinwards.k = 5 not in [-inf, 3] - Forbidden k"),

    (-.5, (-2, 2), (None, 3), "", "\noutwards.y = -0.5 not in [0.0, inf] - Forbidden y"),

    (-2, (-2, 2), (None, 3),
        "\nx_out.x = -1 not in [0, 2] - Reasonable set of value",
        "\noutwards.y = -2 not in [0.0, inf] - Forbidden y"),

    (np.array([3.3, 3]), (-2, 2), (None, 3),
        "\noutwards.y = 6.3 not in [0.5, 3.5] - Reasonable y",
        "\nx_out.x = 7.3 not in [-2, 4] - Don't push your luck\n\tinwards.k = [3.3 3. ] not in [-inf, 3] - Forbidden k"),

    (np.array([3.3, 3]), (None, None), (None, 3),
        "\noutwards.y = 6.3 not in [0.5, 3.5] - Reasonable y",
        "\nx_out.x = 7.3 not in [-2, 4] - Don't push your luck"),

    (np.array([3.3, 3]), None, (None, 3),
        "\noutwards.y = 6.3 not in [0.5, 3.5] - Reasonable y",
        "\nx_out.x = 7.3 not in [-2, 4] - Don't push your luck\n\tinwards.k = [3.3 3. ] not in [-inf, 3] - Forbidden k"),

    (np.array([3.3, 3]), None, None,
        "\noutwards.y = 6.3 not in [0.5, 3.5] - Reasonable y",
        "\nx_out.x = 7.3 not in [-2, 4] - Don't push your luck"),

    (np.array([3.3, 3, 2]), ((-2, 1), (2, 2.5), (0, 4)), ((-5, 5), (-5, 5),(-5, 6)),
        "\ninwards.k = [3.3 3.  2. ] not in [(-2, 1), (2, 2.5), (0, 4)] - Reasonable k\n\toutwards.y = 8.3 not in [0.5, 3.5] - Reasonable y",
        "\nx_out.x = 9.3 not in [-2, 4] - Don't push your luck"),
])
def test_ArrayValidityCheck(caplog, Simple, k_val, k_valid_range, k_limits, warning_msg, error_msg):

    s = Simple("test", k_val, k_valid_range, k_limits)
    s.add_driver(RunOnce("run"))
    s.add_driver(ValidityCheck("check"))

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        s.run_drivers()
        
        assert get_log(caplog, logging.WARNING) == warning_msg 
        assert get_log(caplog, logging.ERROR) == error_msg


@pytest.mark.parametrize("k_val, k_valid_range, k_limits, expected", [
    (
        1, ((2, 3), (2, 6)), ((2, 3), (2, 6)),
        dict(
            error = ValueError,
            match = r"valid_range \(.*\) or limits \(.*\) of variable 'test.inwards.k' are incompatible with its value 1",
        ),
    ),
    (
        np.array([3.3, 3]), 2, (None, 3),
        dict(
            error = TypeError,
            match = "Validity or limit range must be a tuple with format comparable to value",
        ),
    ),
    (
        np.array([3.3, 3]), (1, 2, 3), (None, 3),
        dict(
            error = TypeError,
            match = "Valid range or limits must be a size 2 tuple with type comparable to value",
        ),
    ),
    (
        np.array([3.3, 3]), ((1, 2), (2, 3)), (None, 3),
        dict(
            error = ValueError,
            match = r"valid_range \(.*\) and limits \(None, 3\) of variable 'test.inwards.k' have different formats",
        ),
    ),
    (
        np.array([3.3, 3]), ((1, 2), (2, 3)), ((1, 5), 3),
        dict(
            error = ValueError,
            match = r"Mixed values in valid_range \(\(1, 5\), 3\) of 'test.inwards.k'",
        ),
    ),
])
def test_ArrayValidityException(Simple, k_val, k_valid_range, k_limits, expected):
    error = expected['error']

    with pytest.raises(error, match=expected.get('match', None)):
        s = Simple("test", k_val, k_valid_range, k_limits)
        s.add_driver(RunOnce("run"))
        s.add_driver(ValidityCheck("check"))
        s.run_drivers()
