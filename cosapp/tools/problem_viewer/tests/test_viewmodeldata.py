""" Unit tests for the problem interface."""

import pytest
import unittest
import os

from tempfile import mkdtemp
from collections import OrderedDict

from cosapp.base import System, Port
from cosapp.tools.problem_viewer.problem_viewer import _get_viewer_data, view_model
from cosapp.tests.library.systems.others import ComplexTurbofan


def dict_to_tuples(d: dict):
    return tuple(map(tuple, d.items()))


class TestViewModelData(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        self.dir = mkdtemp()
        self.problem_filename = os.path.join(self.dir, "problem_n2")
        self.problem_html_filename = self.problem_filename
        self.expected_tree = OrderedDict(
            [
                ("name", "turbofan"),
                ("type", "subsystem"),
                ("subsystem_type", "group"),
                (
                    "children",
                    [
                        OrderedDict(
                            [
                                ("name", "atm"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "inlet"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "W_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        )
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "fanC"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "gh"),
                                                                ("type", "param"),
                                                            ]
                                                        )
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "mech_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "XN"),                                                                
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "PW"),                                                                
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "ductC"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "group"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "fl_in"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Tt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Pt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "W",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "merger"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "group",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl1_in",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl2_in",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl_out",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "duct"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "group",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "inwards",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "A",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "cst_loss",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "glp",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl_in",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "outwards",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "PR",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        )
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl_out",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "bleed"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "group",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "inwards",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "split_ratio",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        )
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl_in",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl1_out",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl2_out",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "fl_out"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Tt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Pt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "W",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fan"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "group"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "inwards"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "gh",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        )
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "fl_in"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Tt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Pt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "W",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "mech_in"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "XN",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "PW",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "outwards"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "pcnr",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "pr",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "effis",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "wr",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "PWfan",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "fl_out"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Tt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Pt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "W",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "merger"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "fl1_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl2_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "duct"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "A"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "cst_loss"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "glp"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "outwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "PR"),
                                                                ("type", "unknown"),
                                                            ]
                                                        )
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "bleed"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "split_ratio"),
                                                                ("type", "param"),
                                                            ]
                                                        )
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl1_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl2_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "noz"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Acol"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Aexit"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "outwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "WRnozzle"),
                                                                ("type", "unknown"),
                                                            ]
                                                        )
                                                    ],
                                                ),

                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                    ],
                ),
            ]
        )

        self.expected_conns = [
            dict(src="atm.fl_out.Tt", tgt="inlet.fl_in.Tt"),
            dict(src="atm.fl_out.Pt", tgt="inlet.fl_in.Pt"),
            dict(src="inlet.fl_out.Tt", tgt="fanC.fl_in.Tt"),
            dict(src="inlet.fl_out.Pt", tgt="fanC.fl_in.Pt"),
            dict(src="inlet.fl_out.W", tgt="fanC.fl_in.W"),
            dict(src="fanC.fl_out.Tt", tgt="merger.fl1_in.Tt"),
            dict(src="fanC.fl_out.Pt", tgt="merger.fl1_in.Pt"),
            dict(src="fanC.fl_out.W", tgt="merger.fl1_in.W"),
            dict(src="bleed.fl2_out.Tt", tgt="merger.fl2_in.Tt"),
            dict(src="bleed.fl2_out.Pt", tgt="merger.fl2_in.Pt"),
            dict(src="bleed.fl2_out.W", tgt="merger.fl2_in.W"),
            dict(src="merger.fl_out.Tt", tgt="duct.fl_in.Tt"),
            dict(src="merger.fl_out.Pt", tgt="duct.fl_in.Pt"),
            dict(src="merger.fl_out.W", tgt="duct.fl_in.W"),
            dict(src="duct.fl_out.Tt", tgt="bleed.fl_in.Tt"),
            dict(src="duct.fl_out.Pt", tgt="bleed.fl_in.Pt"),
            dict(src="duct.fl_out.W", tgt="bleed.fl_in.W"),
            dict(src="bleed.fl1_out.Tt", tgt="noz.fl_in.Tt"),
            dict(src="bleed.fl1_out.Pt", tgt="noz.fl_in.Pt"),
            dict(src="bleed.fl1_out.W", tgt="noz.fl_in.W"),
            dict(src="fanC.fl_in.Tt", tgt="fanC.ductC.fl_in.Tt"),
            dict(src="fanC.fl_in.Pt", tgt="fanC.ductC.fl_in.Pt"),
            dict(src="fanC.fl_in.W", tgt="fanC.ductC.fl_in.W"),
            dict(src="fanC.mech_in.XN", tgt="fanC.fan.mech_in.XN"),
            dict(src="fanC.mech_in.PW", tgt="fanC.fan.mech_in.PW"),
            dict(src="fanC.inwards.gh", tgt="fanC.fan.inwards.gh"),
            dict(src="fanC.ductC.fl_out.Tt", tgt="fanC.fan.fl_in.Tt"),
            dict(src="fanC.ductC.fl_out.Pt", tgt="fanC.fan.fl_in.Pt"),
            dict(src="fanC.ductC.fl_out.W", tgt="fanC.fan.fl_in.W"),
            dict(src="fanC.fan.fl_out.Tt", tgt="fanC.fl_out.Tt"),
            dict(src="fanC.fan.fl_out.Pt", tgt="fanC.fl_out.Pt"),
            dict(src="fanC.fan.fl_out.W", tgt="fanC.fl_out.W"),
            dict(src="fanC.ductC.fl_in.Tt", tgt="fanC.ductC.merger.fl1_in.Tt"),
            dict(src="fanC.ductC.fl_in.Pt", tgt="fanC.ductC.merger.fl1_in.Pt"),
            dict(src="fanC.ductC.fl_in.W", tgt="fanC.ductC.merger.fl1_in.W"),
            dict(src="fanC.ductC.bleed.fl2_out.Tt", tgt="fanC.ductC.merger.fl2_in.Tt"),
            dict(src="fanC.ductC.bleed.fl2_out.Pt", tgt="fanC.ductC.merger.fl2_in.Pt"),
            dict(src="fanC.ductC.bleed.fl2_out.W", tgt="fanC.ductC.merger.fl2_in.W"),
            dict(src="fanC.ductC.bleed.fl1_out.Tt", tgt="fanC.ductC.fl_out.Tt"),
            dict(src="fanC.ductC.bleed.fl1_out.Pt", tgt="fanC.ductC.fl_out.Pt"),
            dict(src="fanC.ductC.bleed.fl1_out.W", tgt="fanC.ductC.fl_out.W"),
            dict(src="fanC.ductC.merger.fl_out.Tt", tgt="fanC.ductC.duct.fl_in.Tt"),
            dict(src="fanC.ductC.merger.fl_out.Pt", tgt="fanC.ductC.duct.fl_in.Pt"),
            dict(src="fanC.ductC.merger.fl_out.W", tgt="fanC.ductC.duct.fl_in.W"),
            dict(src="fanC.ductC.duct.fl_out.Tt", tgt="fanC.ductC.bleed.fl_in.Tt"),
            dict(src="fanC.ductC.duct.fl_out.Pt", tgt="fanC.ductC.bleed.fl_in.Pt"),
            dict(src="fanC.ductC.duct.fl_out.W", tgt="fanC.ductC.bleed.fl_in.W"),
        ]

        self.expected_tree_default = OrderedDict(
            [
                ("name", "turbofan"),
                ("type", "subsystem"),
                ("subsystem_type", "group"),
                (
                    "children",
                    [
                        OrderedDict(
                            [
                                ("name", "atm"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                ("children", []),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "inlet"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "W_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        )
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "fanC"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                ("children", []),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "mech_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "XN"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "PW"),                                                                
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "ductC"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "group"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "fl_in"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Tt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Pt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "W",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "merger"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "group",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl1_in",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl2_in",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl_out",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "duct"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "group",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "inwards",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl_in",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "outwards",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl_out",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "bleed"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "group",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "inwards",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl_in",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "param",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl1_out",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "fl2_out",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "subsystem",
                                                                                ),
                                                                                (
                                                                                    "subsystem_type",
                                                                                    "component",
                                                                                ),
                                                                                (
                                                                                    "children",
                                                                                    [
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Tt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "Pt",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                        OrderedDict(
                                                                                            [
                                                                                                (
                                                                                                    "name",
                                                                                                    "W",
                                                                                                ),
                                                                                                (
                                                                                                    "type",
                                                                                                    "unknown",
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ],
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "fl_out"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Tt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Pt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "W",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fan"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "group"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "inwards"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                ("children", []),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "fl_in"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Tt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Pt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "W",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "mech_in"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "XN",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "PW",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "param",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "outwards"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                ("children", []),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "fl_out"),
                                                                ("type", "subsystem"),
                                                                (
                                                                    "subsystem_type",
                                                                    "component",
                                                                ),
                                                                (
                                                                    "children",
                                                                    [
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Tt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "Pt",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                        OrderedDict(
                                                                            [
                                                                                (
                                                                                    "name",
                                                                                    "W",
                                                                                ),
                                                                                (
                                                                                    "type",
                                                                                    "unknown",
                                                                                ),
                                                                            ]
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "merger"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "fl1_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl2_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "duct"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                ("children", []),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "outwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                ("children", []),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "bleed"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                ("children", []),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl1_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl2_out"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "unknown"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        OrderedDict(
                            [
                                ("name", "noz"),
                                ("type", "subsystem"),
                                ("subsystem_type", "group"),
                                (
                                    "children",
                                    [
                                        OrderedDict(
                                            [
                                                ("name", "inwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                ("children", []),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "fl_in"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                (
                                                    "children",
                                                    [
                                                        OrderedDict(
                                                            [
                                                                ("name", "Tt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "Pt"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                        OrderedDict(
                                                            [
                                                                ("name", "W"),
                                                                ("type", "param"),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        OrderedDict(
                                            [
                                                ("name", "outwards"),
                                                ("type", "subsystem"),
                                                ("subsystem_type", "component"),
                                                ("children", []),
                                            ]
                                        ),
                                    ],
                                ),
                            ]
                        ),
                    ],
                ),
            ]
        )

        self.expected_conns_default = [
            dict(src="atm.fl_out.Tt", tgt="inlet.fl_in.Tt"),
            dict(src="atm.fl_out.Pt", tgt="inlet.fl_in.Pt"),
            dict(src="inlet.fl_out.Tt", tgt="fanC.fl_in.Tt"),
            dict(src="inlet.fl_out.Pt", tgt="fanC.fl_in.Pt"),
            dict(src="inlet.fl_out.W", tgt="fanC.fl_in.W"),
            dict(src="fanC.fl_out.Tt", tgt="merger.fl1_in.Tt"),
            dict(src="fanC.fl_out.Pt", tgt="merger.fl1_in.Pt"),
            dict(src="fanC.fl_out.W", tgt="merger.fl1_in.W"),
            dict(src="bleed.fl2_out.Tt", tgt="merger.fl2_in.Tt"),
            dict(src="bleed.fl2_out.Pt", tgt="merger.fl2_in.Pt"),
            dict(src="bleed.fl2_out.W", tgt="merger.fl2_in.W"),
            dict(src="merger.fl_out.Tt", tgt="duct.fl_in.Tt"),
            dict(src="merger.fl_out.Pt", tgt="duct.fl_in.Pt"),
            dict(src="merger.fl_out.W", tgt="duct.fl_in.W"),
            dict(src="duct.fl_out.Tt", tgt="bleed.fl_in.Tt"),
            dict(src="duct.fl_out.Pt", tgt="bleed.fl_in.Pt"),
            dict(src="duct.fl_out.W", tgt="bleed.fl_in.W"),
            dict(src="bleed.fl1_out.Tt", tgt="noz.fl_in.Tt"),
            dict(src="bleed.fl1_out.Pt", tgt="noz.fl_in.Pt"),
            dict(src="bleed.fl1_out.W", tgt="noz.fl_in.W"),
            dict(src="fanC.fl_in.Tt", tgt="fanC.ductC.fl_in.Tt"),
            dict(src="fanC.fl_in.Pt", tgt="fanC.ductC.fl_in.Pt"),
            dict(src="fanC.fl_in.W", tgt="fanC.ductC.fl_in.W"),
            dict(src="fanC.mech_in.XN", tgt="fanC.fan.mech_in.XN"),
            dict(src="fanC.mech_in.PW", tgt="fanC.fan.mech_in.PW"),
            dict(src="fanC.ductC.fl_out.Tt", tgt="fanC.fan.fl_in.Tt"),
            dict(src="fanC.ductC.fl_out.Pt", tgt="fanC.fan.fl_in.Pt"),
            dict(src="fanC.ductC.fl_out.W", tgt="fanC.fan.fl_in.W"),
            dict(src="fanC.fan.fl_out.Tt", tgt="fanC.fl_out.Tt"),
            dict(src="fanC.fan.fl_out.Pt", tgt="fanC.fl_out.Pt"),
            dict(src="fanC.fan.fl_out.W", tgt="fanC.fl_out.W"),
            dict(src="fanC.ductC.fl_in.Tt", tgt="fanC.ductC.merger.fl1_in.Tt"),
            dict(src="fanC.ductC.fl_in.Pt", tgt="fanC.ductC.merger.fl1_in.Pt"),
            dict(src="fanC.ductC.fl_in.W", tgt="fanC.ductC.merger.fl1_in.W"),
            dict(src="fanC.ductC.bleed.fl2_out.Tt", tgt="fanC.ductC.merger.fl2_in.Tt"),
            dict(src="fanC.ductC.bleed.fl2_out.Pt", tgt="fanC.ductC.merger.fl2_in.Pt"),
            dict(src="fanC.ductC.bleed.fl2_out.W", tgt="fanC.ductC.merger.fl2_in.W"),
            dict(src="fanC.ductC.bleed.fl1_out.Tt", tgt="fanC.ductC.fl_out.Tt"),
            dict(src="fanC.ductC.bleed.fl1_out.Pt", tgt="fanC.ductC.fl_out.Pt"),
            dict(src="fanC.ductC.bleed.fl1_out.W", tgt="fanC.ductC.fl_out.W"),
            dict(src="fanC.ductC.merger.fl_out.Tt", tgt="fanC.ductC.duct.fl_in.Tt"),
            dict(src="fanC.ductC.merger.fl_out.Pt", tgt="fanC.ductC.duct.fl_in.Pt"),
            dict(src="fanC.ductC.merger.fl_out.W", tgt="fanC.ductC.duct.fl_in.W"),
            dict(src="fanC.ductC.duct.fl_out.Tt", tgt="fanC.ductC.bleed.fl_in.Tt"),
            dict(src="fanC.ductC.duct.fl_out.Pt", tgt="fanC.ductC.bleed.fl_in.Pt"),
            dict(src="fanC.ductC.duct.fl_out.W", tgt="fanC.ductC.bleed.fl_in.W"),
        ]

    @staticmethod
    def make_system():
        return ComplexTurbofan("turbofan")

    def test_model_viewer_has_correct_data_from_problem(self):
        """
        Verify that the correct model structure data exists when stored as compared
        to the expected structure, using the SellarStateConnection model.
        """
        p = self.make_system()

        model_viewer_data = _get_viewer_data(p, include_orphan_vars=True)

        assert model_viewer_data["tree"] == self.expected_tree
        # Check connection list, regardless of order
        actual = set(map(dict_to_tuples, model_viewer_data["connections_list"]))
        expected = set(map(dict_to_tuples, self.expected_conns))
        assert actual == expected

    def test_model_viewer_takes_into_account_include_wards(self):
        """
        Test that model viewer does not show orphan vars by default.
        """
        p = self.make_system()

        model_viewer_data = _get_viewer_data(p, include_orphan_vars=False)
        # tree_json = model_viewer_data["tree"]
        # conns_json = model_viewer_data["connections_list"]

        assert model_viewer_data["tree"] == self.expected_tree_default
        # Check connection list, regardless of order
        actual = set(map(dict_to_tuples, model_viewer_data["connections_list"]))
        expected = set(map(dict_to_tuples, self.expected_conns_default))
        assert actual == expected

    def test_view_model_from_problem(self):
        """
        Test that an n2 html file is generated from a Problem.
        """
        p = self.make_system()

        view_model(p, outfile=self.problem_filename, show_browser=False)

        # Check that the html file has been created and has something in it.
        assert os.path.isfile(self.problem_html_filename),  f"{self.problem_html_filename} is not a valid file."
        assert os.path.getsize(self.problem_html_filename) > 100


def test_get_viewer_data_same_name():
    """Check that no confusion exists in connectors
    when a subsystem and its parent have the same name.
    """
    class XyPort(Port):
        def setup(self):
            self.add_variable("x", 1.0)
            self.add_variable("y", 1.0)

    class Bogus(System):
        def setup(self):
            self.add_input(XyPort, 'p_in')
            self.add_output(XyPort, 'p_out')
    
    top = System("top")
    a = Bogus("top")  # same name as parent
    b = Bogus("b")
    top.add_child(a)
    top.add_child(b, pulling='p_out')  # connector to `top.fl_out`
    top.connect(a.p_in, b.p_out)  # connector to `top.top.fl_out`
    top.exec_order = ['top', 'b']

    assert set(top.child_connectors) == {'top'}
    assert set(top.connectors()) == {
        'b.p_out -> p_out',
        'b.p_out -> top.p_in',
    }

    data = _get_viewer_data(top, True)
    expected_connections = [
        {'src': 'b.p_out.x', 'tgt': 'p_out.x'},
        {'src': 'b.p_out.y', 'tgt': 'p_out.y'},
        {'src': 'b.p_out.x', 'tgt': 'top.p_in.x'},
        {'src': 'b.p_out.y', 'tgt': 'top.p_in.y'},
    ]
    # Check connection list, regardless of order
    actual_pairs = set(map(dict_to_tuples, data["connections_list"]))
    expected_pairs = set(map(dict_to_tuples, expected_connections))
    assert actual_pairs == expected_pairs
