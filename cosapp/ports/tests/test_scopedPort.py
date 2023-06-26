import pytest
import numpy
from unittest import mock

from cosapp.ports.exceptions import ScopeError
from cosapp.ports import Scope
from cosapp.systems import System
from cosapp.drivers import NonLinearSolver, RunSingleCase
from cosapp.tests.library.systems import NonLinear1, Multiply2, MergerMath, SplitterMath
from cosapp.tests.library.ports import XPort
from typing import List, FrozenSet


def make_roles(role_lists: List[List[str]]) -> FrozenSet[FrozenSet[str]]:
    """Utility function to format lists of roles into
    nested frozensets, as expected by `CoSAppConfiguration`.
    """
    return frozenset(
        map(frozenset, role_lists)
    )


class PublicSystem(System):

    def setup(self):
        self.add_inward("a", 1.0, scope=Scope.PRIVATE)
        self.add_inward("b", 2.0, scope=Scope.PROTECTED)
        self.add_inward("c", 3.0, scope=Scope.PUBLIC)

        self.add_outward("v", numpy.zeros(3))

    def compute(self):
        self.v[:] = [self.a, self.b, self.c]


class ScopedSystem(PublicSystem):

    tags = ["privacy", "protection"]

    def setup(self):
        super().setup()

        self.add_design_method("design_a").add_unknown("a").add_equation("v[0] == 0")
        self.add_design_method("design_b").add_unknown("b").add_equation("v[1] == 0")


def test_make_roles():
    """Sanity check on utility function `make_roles`"""
    roles = [
        ["apple", "cherry", "apple"],
        ["privacy", "extra", "protection"],
    ]
    assert make_roles(roles) == frozenset([
        frozenset(["apple", "cherry"]),
        frozenset(["privacy", "extra", "protection"]),
    ])
    assert make_roles([]) == frozenset()


@mock.patch("cosapp.core.config.CoSAppConfiguration")
def test_ScopedPort_system_changing_own_var(Config):
    Config.userid = "id1234"
    Config().roles = frozenset()

    # Public user

    s = PublicSystem("dummy")
    # Test set operations
    s.a = -1.
    s.b = -2.
    s.c = -3.
    s.run_once()
    assert numpy.array_equal(s.v, [-1, -2, -3])

    s = ScopedSystem("dummy")
    # Test set operations
    with pytest.raises(ScopeError):
        s.a = -1.
    with pytest.raises(ScopeError):
        s.b = -2.
    s.c = -3.
    s.run_once()
    assert numpy.array_equal(s.v, [1, 2, -3])

    # Protected user
    Config().roles = make_roles([["protection"]])

    s = ScopedSystem("dummy")
    # Test set operations
    with pytest.raises(ScopeError):
        s.a = -1.
    s.b = -2.
    s.c = -3.
    s.run_once()
    assert numpy.array_equal(s.v, [1, -2, -3])

    # Private user
    Config().roles = make_roles([["privacy", "protection"]])

    s = ScopedSystem("dummy")
    # Test set operations
    s.a = -1.
    s.b = -2.
    s.c = -3.
    s.run_once()
    assert numpy.array_equal(s.v, [-1, -2, -3])

    # Private user, with extra roles, compared to tags
    Config().roles = make_roles(
        [
            ["apple", "cherry"],
            ["privacy", "extra", "protection"],
        ]
    )
    s = ScopedSystem("dummy")
    # Test set operations
    s.a = -1.
    s.b = -2.
    s.c = -3.
    s.run_once()
    assert numpy.array_equal(s.v, [-1, -2, -3])

    # Private user, with incomplete role and fully matching role
    Config().roles = make_roles(
        [
            ["apple", "cherry"],
            ["privacy"],
            ["privacy", "extra", "protection"],  # should prevail
        ]
    )
    s = ScopedSystem("dummy")
    # Test set operations
    s.a = -1.
    s.b = -2.
    s.c = -3.
    s.run_once()
    assert numpy.array_equal(s.v, [-1, -2, -3])


@mock.patch("cosapp.core.config.CoSAppConfiguration")
def test_ScopedPort_solver_changing_unknown(Config):
    # A solver should be able to modify a "PRIVATE" or "PROTECTED" unknown local
    # but not a global (i.e. an external user cannot set as unknown a hidden variable)

    def build_case(splitter_type):
        class Head(System):
            def setup(self):
                self.add_child(NonLinear1("nonlinear"))
                self.add_child(Multiply2("mult2"))
                self.add_child(MergerMath("merger"), pulling={"p1_in": "p_in"})
                self.add_child(splitter_type("splitter"), pulling={"p2_out": "p_out"})

                self.connect(self.nonlinear.p_out, self.merger.p2_in)
                self.connect(self.merger.p_out, self.mult2.p_in)
                self.connect(self.mult2.p_out, self.splitter.p_in)
                self.connect(self.splitter.p1_out, self.nonlinear.p_in)

                self.exec_order = ["merger", "mult2", "splitter", "nonlinear"]

        s = Head("s")

        s.mult2.K1 = 1
        s.mult2.K2 = 1
        s.nonlinear.k1 = 1
        s.nonlinear.k2 = 0.5

        solver = s.add_driver(NonLinearSolver("solver", factor=0.05))
        runner = solver.add_child(RunSingleCase("runner"))

        solver.add_unknown(
            ["mult2.K1", "nonlinear.k2"]
        )
        solver.add_equation([
            "splitter.p2_out.x == 50.",
            "splitter.p1_out.x == 5.",
            "merger.p_out.x == 30.",
        ])
        runner.set_values({
            "p_in.x": 10,
        })

        return s

    # No tags case => everything can be changed
    s = build_case(SplitterMath)
    s.drivers["solver"].add_unknown("splitter.split_ratio")
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.86136, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.090909091, rel=1e-3)

    # Public iterative
    #  Globally unfrozen => no error
    class PublicFrozenSplitter(System):

        tags = ["privacy", "protection"]

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PUBLIC)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    s = build_case(PublicFrozenSplitter)
    s.drivers["solver"].add_unknown("splitter.split_ratio")
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.86136, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.090909091, rel=1e-3)

    # Protected iterative
    #  Locally unfrozen => no error
    class ProtectedFreeSplitter(System):

        tags = ["privacy", "protection"]

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PROTECTED)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

            self.add_unknown("split_ratio")

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    s = build_case(ProtectedFreeSplitter)
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.86136, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.090909091, rel=1e-3)

    #  Globally unfrozen => error for common user
    class ProtectedFrozenSplitter(System):

        tags = ["privacy", "protection"]

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PROTECTED)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    s = build_case(ProtectedFrozenSplitter)
    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        s.drivers["solver"].add_unknown("splitter.split_ratio")

    #  Globally unfrozen => no error for protected user
    Config().roles = make_roles([["protection"]])
    ProtectedFrozenSplitter._user_context = None
    s = build_case(ProtectedFrozenSplitter)
    s.drivers["solver"].add_unknown("splitter.split_ratio")
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.86136, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.090909091, rel=1e-3)

    #  Globally unfrozen => no error for private user
    Config().roles = make_roles([["privacy", "protection"]])
    ProtectedFrozenSplitter._user_context = None
    s = build_case(ProtectedFrozenSplitter)
    s.drivers["solver"].add_unknown("splitter.split_ratio")
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.86136, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.090909091, rel=1e-3)

    # Private iterative
    #  Locally unfrozen => no error
    class PrivateFreeSplitter(System):

        tags = ["privacy", "protection"]

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PRIVATE)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

            self.add_unknown("split_ratio")

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    s = build_case(PrivateFreeSplitter)
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.86136, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.090909091, rel=1e-3)

    #  Globally unfrozen => error for common user
    Config().roles = frozenset()

    class PrivateFrozenSplitter(System):

        tags = ["privacy", "protection"]

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PRIVATE)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    s = build_case(PrivateFrozenSplitter)
    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        s.drivers["solver"].add_unknown("splitter.split_ratio")

    #  Globally unfrozen => error for protected user
    Config().roles = make_roles([["protection"]])
    PrivateFrozenSplitter._user_context = None
    s = build_case(PrivateFrozenSplitter)
    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        s.drivers["solver"].add_unknown("splitter.split_ratio")

    #  Globally unfrozen => no error for private user
    Config().roles = make_roles([["privacy", "protection"]])
    PrivateFrozenSplitter._user_context = None
    s = build_case(PrivateFrozenSplitter)
    s.drivers["solver"].add_unknown("splitter.split_ratio")
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.86136, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.090909091, rel=1e-3)


@mock.patch("cosapp.core.config.CoSAppConfiguration")
def test_ScopedPort_driver_set_boundary(Config):
    # A driver should not be able to modify a "PRIVATE" or "PROTECTED" variable

    def build_case(nl_type):
        class Head(System):
            def setup(self):
                self.add_child(nl_type("nonlinear"))
                self.add_child(Multiply2("mult2"))
                self.add_child(MergerMath("merger"), pulling={"p1_in": "p_in"})
                self.add_child(SplitterMath("splitter"), pulling={"p2_out": "p_out"})

                self.connect(self.nonlinear.p_out, self.merger.p2_in)
                self.connect(self.merger.p_out, self.mult2.p_in)
                self.connect(self.mult2.p_out, self.splitter.p_in)
                self.connect(self.splitter.p1_out, self.nonlinear.p_in)

                self.exec_order = ["merger", "mult2", "splitter", "nonlinear"]

        s = Head("s")

        s.mult2.K1 = 1
        s.mult2.K2 = 1

        solver = s.add_driver(NonLinearSolver("solver", factor=0.05))
        runner = solver.add_child(RunSingleCase("runner"))

        solver.add_unknown(
            ["mult2.K1", "splitter.split_ratio"]
        )
        solver.add_equation([
            "splitter.p2_out.x == 50.",
            "splitter.p1_out.x == 5.",
            "merger.p_out.x == 30.",
        ])
        runner.set_values({
            "p_in.x": 10.0,
            # This is the boundary for testing
            "nonlinear.k1": 2,
        })

        return s

    # No tags case => everything can be changed
    s = build_case(NonLinear1)
    s.drivers["solver"].add_unknown("nonlinear.k2")
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.43068, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.09091, rel=1e-3)

    # Public iterative
    #  Set boundary => no error
    Config().roles = frozenset()

    class PublicNonLinear1(System):

        tags = ["privacy", "protection"]

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("k1", 5.0, scope=Scope.PUBLIC)
            self.add_inward("k2", 5.0)
            self.add_output(XPort, "p_out", {"x": 1.0})

            self.add_unknown("k2")

        def compute(self):
            self.p_out.x = self.k1 * self.p_in.x ** self.k2

    s = build_case(PublicNonLinear1)
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.43068, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.09091, rel=1e-3)

    # Protected iterative
    #  Set boundary => error
    Config().roles = frozenset()

    class ProtectedNonLinear1(System):

        tags = ["privacy", "protection"]

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("k1", 5.0, scope=Scope.PROTECTED)
            self.add_inward("k2", 5.0)
            self.add_output(XPort, "p_out", {"x": 1.0})

            self.add_unknown("k2")

        def compute(self):
            self.p_out.x = self.k1 * self.p_in.x ** self.k2

    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        build_case(ProtectedNonLinear1)

    #  Set boundary => no error for protected user
    Config().roles = make_roles([["protection"]])
    ProtectedNonLinear1._user_context = None
    s = build_case(ProtectedNonLinear1)
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.43068, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.09091, rel=1e-3)

    #  Set boundary => no error for private user
    Config().roles = make_roles([["privacy", "protection"]])
    ProtectedNonLinear1._user_context = None
    s = build_case(ProtectedNonLinear1)
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.43068, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.09091, rel=1e-3)

    # Private iterative
    #  Set boundary => error for common user
    Config().roles = frozenset()

    class PrivateNonLinear1(System):

        tags = ["privacy", "protection"]

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("k1", 5.0, scope=Scope.PRIVATE)
            self.add_inward("k2", 5.0)
            self.add_output(XPort, "p_out", {"x": 1.0})

            self.add_unknown("k2")

        def compute(self):
            self.p_out.x = self.k1 * self.p_in.x ** self.k2

    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        build_case(PrivateNonLinear1)

    #  Set boundary => error for protected user
    Config().roles = make_roles([["protection"]])
    PrivateNonLinear1._user_context = None

    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        build_case(PrivateNonLinear1)

    #  Set boundary => no error for private user
    Config().roles = make_roles([["privacy", "protection"]])
    PrivateNonLinear1._user_context = None
    s = build_case(PrivateNonLinear1)
    s.run_drivers()

    assert s.mult2.K1 == pytest.approx(1.83333, rel=1e-3)
    assert s.nonlinear.k2 == pytest.approx(1.43068, rel=1e-3)
    assert s.splitter.split_ratio == pytest.approx(0.09091, rel=1e-3)
