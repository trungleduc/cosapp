import pytest
from unittest import mock

from cosapp.ports.exceptions import ScopeError
from cosapp.ports import Scope
from cosapp.systems import System
from cosapp.drivers import NonLinearSolver
from cosapp.tests.library.systems import NonLinear1, Multiply2, MergerMath, SplitterMath
from cosapp.tests.library.ports import XPort


class ScopedS(System):
    tags = frozenset(["privacy", "protection"])

    def setup(self):
        self.add_inward("A", 1.0, scope=Scope.PRIVATE)
        self.add_inward("B", 2.0, scope=Scope.PROTECTED)
        self.add_inward("C", 3.0, scope=Scope.PUBLIC)

        self.add_design_method("design_a").add_unknown("A").add_equation("A == 0")
        self.add_design_method("design_b").add_unknown("B").add_equation("B == 0")

    def compute(self):
        self.A = 2.0
        self.B = 4.0
        self.C = 6.0


class PublicS(System):
    def setup(self):
        self.add_inward("A", 1.0, scope=Scope.PRIVATE)
        self.add_inward("B", 2.0, scope=Scope.PROTECTED)
        self.add_inward("C", 3.0, scope=Scope.PUBLIC)

    def compute(self):
        self.A = 2.0
        self.B = 4.0
        self.C = 6.0


@mock.patch("cosapp.core.config.CoSAppConfiguration")
def test_ScopedPort_system_changing_own_var(Config):
    Config.userid = "id1234"
    Config().roles = frozenset()

    # Public user

    s = PublicS("dummy")
    # Test set operations
    s.A = -1
    s.B = -2
    s.C = -3
    s.run_once()
    assert s.A == 2.0
    assert s.B == 4.0
    assert s.C == 6.0

    s = ScopedS("dummy")
    # Test set operations
    with pytest.raises(ScopeError):
        s.A = -1
    with pytest.raises(ScopeError):
        s.B = -2
    s.C = -3
    s.run_once()
    assert s.A == 2.0
    assert s.B == 4.0
    assert s.C == 6.0

    # Protected user
    Config().roles = frozenset([frozenset(["protection"])])

    s = ScopedS("dummy")
    # Test set operations
    with pytest.raises(ScopeError):
        s.A = -1
    s.B = -2
    s.C = -3
    s.run_once()
    assert s.A == 2.0
    assert s.B == 4.0
    assert s.C == 6.0

    # Private user
    Config().roles = frozenset([frozenset(["privacy", "protection"])])

    s = ScopedS("dummy")
    # Test set operations
    s.A = -1
    s.B = -2
    s.C = -3
    s.run_once()
    assert s.A == 2.0
    assert s.B == 4.0
    assert s.C == 6.0


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

        snl = Head("nl")

        design = snl.add_driver(NonLinearSolver("design", factor=0.05))

        snl.mult2.inwards.K1 = 1
        snl.mult2.inwards.K2 = 1
        snl.nonlinear.inwards.k1 = 1
        snl.nonlinear.inwards.k2 = 0.5

        design.runner.set_values({"p_in.x": 10})
        design.runner.design.add_unknown(
            ["mult2.inwards.K1", "nonlinear.inwards.k2"]
        ).add_equation(
            [
                "splitter.p2_out.x == 50.",
                "merger.p_out.x == 30.",
                "splitter.p1_out.x == 5.",
            ]
        )

        return snl

    # No tags case => everything can be changed
    snl = build_case(SplitterMath)
    snl.drivers["design"].runner.design.add_unknown("splitter.inwards.split_ratio")
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.86136, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.090909091, abs=10e-4)

    # Public iterative
    #  Globally unfrozen => no error
    class PublicFrozenSplitter(System):
        tags = frozenset(["privacy", "protection"])

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PUBLIC)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    snl = build_case(PublicFrozenSplitter)
    snl.drivers["design"].runner.design.add_unknown("splitter.inwards.split_ratio")
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.86136, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.090909091, abs=10e-4)

    # Protected iterative
    #  Locally unfrozen => no error
    class ProtectedFreeSplitter(System):
        tags = frozenset(["privacy", "protection"])

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PROTECTED)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

            self.add_unknown("split_ratio")

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    snl = build_case(ProtectedFreeSplitter)
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.86136, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.090909091, abs=10e-4)

    #  Globally unfrozen => error for common user
    class ProtectedFrozenSplitter(System):
        tags = frozenset(["privacy", "protection"])

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PROTECTED)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    snl = build_case(ProtectedFrozenSplitter)
    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        snl.drivers["design"].runner.design.add_unknown("splitter.inwards.split_ratio")

    #  Globally unfrozen => no error for protected user
    Config().roles = frozenset([frozenset(["protection"])])
    ProtectedFrozenSplitter._user_context = None
    snl = build_case(ProtectedFrozenSplitter)
    snl.drivers["design"].runner.design.add_unknown("splitter.inwards.split_ratio")
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.86136, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.090909091, abs=10e-4)

    #  Globally unfrozen => no error for private user
    Config().roles = frozenset([frozenset(["privacy", "protection"])])
    ProtectedFrozenSplitter._user_context = None
    snl = build_case(ProtectedFrozenSplitter)
    snl.drivers["design"].runner.design.add_unknown("splitter.inwards.split_ratio")
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.86136, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.090909091, abs=10e-4)

    # Private iterative
    #  Locally unfrozen => no error
    class PrivateFreeSplitter(System):
        tags = frozenset(["privacy", "protection"])

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PRIVATE)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

            self.add_unknown("split_ratio")

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    snl = build_case(PrivateFreeSplitter)
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.86136, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.090909091, abs=10e-4)

    #  Globally unfrozen => error for common user
    Config().roles = frozenset()

    class PrivateFrozenSplitter(System):
        tags = frozenset(["privacy", "protection"])

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("split_ratio", 0.1, scope=Scope.PRIVATE)
            self.add_output(XPort, "p1_out", {"x": 1.0})
            self.add_output(XPort, "p2_out", {"x": 1.0})

        def compute(self):
            self.p1_out.x = self.p_in.x * self.split_ratio
            self.p2_out.x = self.p_in.x * (1 - self.split_ratio)

    snl = build_case(PrivateFrozenSplitter)
    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        snl.drivers["design"].runner.design.add_unknown("splitter.inwards.split_ratio")

    #  Globally unfrozen => error for protected user
    Config().roles = frozenset([frozenset(["protection"])])
    PrivateFrozenSplitter._user_context = None
    snl = build_case(PrivateFrozenSplitter)
    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        snl.drivers["design"].runner.design.add_unknown("splitter.inwards.split_ratio")

    #  Globally unfrozen => no error for private user
    Config().roles = frozenset([frozenset(["privacy", "protection"])])
    PrivateFrozenSplitter._user_context = None
    snl = build_case(PrivateFrozenSplitter)
    snl.drivers["design"].runner.design.add_unknown("splitter.inwards.split_ratio")
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.86136, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.090909091, abs=10e-4)


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

        snl = Head("nl")

        design = snl.add_driver(NonLinearSolver("design", factor=0.05))

        snl.mult2.inwards.K1 = 1
        snl.mult2.inwards.K2 = 1

        design.runner.set_values(
            {
                "p_in.x": 10.0,
                # This is the boundary for testing
                "nonlinear.inwards.k1": 2,
            }
        )
        design.runner.design.add_unknown(
            ["mult2.inwards.K1", "splitter.inwards.split_ratio"]
        ).add_equation(
            [
                "splitter.p2_out.x == 50.",
                "merger.p_out.x == 30.",
                "splitter.p1_out.x == 5.",
            ]
        )

        return snl

    # No tags case => everything can be changed
    snl = build_case(NonLinear1)
    snl.drivers["design"].runner.design.add_unknown("nonlinear.inwards.k2")
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.43068, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.09091, abs=10e-4)

    # Public iterative
    #  Set boundary => no error
    Config().roles = frozenset()

    class PublicNonLinear1(System):
        tags = frozenset(["privacy", "protection"])

        def setup(self):
            self.add_input(XPort, "p_in", {"x": 1.0})
            self.add_inward("k1", 5.0, scope=Scope.PUBLIC)
            self.add_inward("k2", 5.0)
            self.add_output(XPort, "p_out", {"x": 1.0})

            self.add_unknown("k2")

        def compute(self):
            self.p_out.x = self.k1 * self.p_in.x ** self.k2

    snl = build_case(PublicNonLinear1)
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.43068, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.09091, abs=10e-4)

    # Protected iterative
    #  Set boundary => error
    Config().roles = frozenset()

    class ProtectedNonLinear1(System):
        tags = frozenset(["privacy", "protection"])

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
    Config().roles = frozenset([frozenset(["protection"])])
    ProtectedNonLinear1._user_context = None
    snl = build_case(ProtectedNonLinear1)
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.43068, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.09091, abs=10e-4)

    #  Set boundary => no error for private user
    Config().roles = frozenset([frozenset(["privacy", "protection"])])
    ProtectedNonLinear1._user_context = None
    snl = build_case(ProtectedNonLinear1)
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.43068, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.09091, abs=10e-4)

    # Private iterative
    #  Set boundary => error for common user
    Config().roles = frozenset()

    class PrivateNonLinear1(System):
        tags = frozenset(["privacy", "protection"])

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
    Config().roles = frozenset([frozenset(["protection"])])
    PrivateNonLinear1._user_context = None

    with pytest.raises(
        ScopeError,
        match=r"Trying to set variable '\w+[\.\w+]*' out of your scope through a boundary\.",
    ):
        build_case(PrivateNonLinear1)

    #  Set boundary => no error for private user
    Config().roles = frozenset([frozenset(["privacy", "protection"])])
    PrivateNonLinear1._user_context = None
    snl = build_case(PrivateNonLinear1)
    snl.run_drivers()

    assert snl.mult2.inwards.K1 == pytest.approx(1.83333, abs=10e-4)
    assert snl.nonlinear.inwards.k2 == pytest.approx(1.43068, abs=10e-4)
    assert snl.splitter.inwards.split_ratio == pytest.approx(0.09091, abs=10e-4)
