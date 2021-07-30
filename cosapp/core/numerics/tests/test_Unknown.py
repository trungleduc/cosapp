import pytest
import numpy as np
import logging, re

from cosapp.systems import System
from cosapp.drivers import NonLinearSolver, NonLinearMethods, RunSingleCase
from cosapp.core.numerics.boundary import Unknown
from cosapp.ports import Port


class TestUnknown:
    class APort(Port):
        def setup(self):
            self.add_variable('m')

    class ASyst(System):
        def setup(self, **kwargs):
            self.add_input(TestUnknown.APort, 'in_')
            self.add_inward('x')
            self.add_inward('y')
            self.add_inward('v', np.zeros(3))
            self.add_outward('out')


    @pytest.mark.parametrize("name, options, expected", [
        ('in_.m', dict(), dict(port='in_')),
        ('x', dict(), dict(port='inwards', name='inwards.x')),
        ('in_.m', dict(upper_bound=2., max_rel_step=0.001), dict(port='in_')),
        ('x', dict(upper_bound=10, max_rel_step=0.05), dict(port='inwards', name='inwards.x')),
        ('y', dict(lower_bound=-30, max_abs_step=1.2), dict(port='inwards', name='inwards.y')),
        ('v', dict(), dict(port='inwards', name='inwards.v', mask=[True, True, True])),
        ('v', dict(lower_bound=0), dict(port='inwards', name='inwards.v', mask=[True, True, True], lower_bound=0)),
        ('v[::2]', dict(), dict(port='inwards', name='inwards.v', mask=[True, False, True])),
        ('v[:-1]', dict(), dict(port='inwards', name='inwards.v', mask=[True, True, False])),
        ('v', dict(lower_bound=[0, 1, -np.inf]), dict(error=TypeError, match="'lower_bound' should be Number")),
        ('v', dict(upper_bound=[0, 1, 3]), dict(error=TypeError, match="'upper_bound' should be Number")),
        # ('x', dict(lower_bound=np.inf), dict(error=ValueError)),   # Bug?
        # ('x', dict(upper_bound=-np.inf), dict(error=ValueError)),   # Bug?
        ('x', dict(lower_bound='a'), dict(error=TypeError)),
        ('x', dict(upper_bound='a'), dict(error=TypeError)),
        ('x', dict(max_abs_step='a'), dict(error=TypeError)),
        ('x', dict(max_rel_step='a'), dict(error=TypeError)),
        ('x', dict(max_rel_step=0), dict(error=ValueError)),
        ('x', dict(max_abs_step=0), dict(error=ValueError)),
        ('out', dict(), dict(error=ValueError, match="Only variables in input ports can be used as boundaries")),
    ])
    def test___init__(self, name, options, expected):
        def get_expected(key, default=None):
            """Set expected[key] to options[key] if `key` is not specified in `expected`.
            Use `default` if all else fails. Returns expected[key]."""
            return expected.setdefault(key, options.get(key, default))
        system = TestUnknown.ASyst('a')
        error = expected.get('error', None)
        if error is None:
            unknown = Unknown(system, name, **options)
            assert unknown.context is system
            assert unknown.name == expected.get('name', name)
            assert unknown.port is system[expected['port']]
            assert unknown.lower_bound == get_expected('lower_bound', -np.inf)
            assert unknown.upper_bound == get_expected('upper_bound', np.inf)
            assert unknown.max_abs_step == get_expected('max_abs_step', np.inf)
            assert unknown.max_rel_step == get_expected('max_rel_step', np.inf)
            np.testing.assert_array_equal(unknown.mask, expected.get('mask', None))
        else:
            pattern = expected.get('match', None)
            with pytest.raises(error, match=pattern):
                Unknown(system, name, **options)


    def test_copy(self):
        a = TestUnknown.ASyst('a')
        original = Unknown(a, 'in_.m', upper_bound=2, max_rel_step=0.001)
        copy = original.copy()
        assert copy is not original
        assert copy.context is original.context
        for attr in ('name', 'port', 'lower_bound', 'upper_bound', 'max_abs_step', 'max_rel_step'):
            assert getattr(copy, attr) == getattr(original, attr)
        np.testing.assert_array_equal(copy.mask, original.mask)


    @pytest.mark.parametrize("name, options, expected", [
        ('in_.m', dict(), dict(port='in_')),
        ('x', dict(), dict(port='inwards', name='inwards.x')),
        ('in_.m', dict(upper_bound=2., max_rel_step=0.001), dict(port='in_')),
        ('x', dict(upper_bound=10, max_rel_step=0.05), dict(port='inwards', name='inwards.x')),
        ('y', dict(lower_bound=-30, max_abs_step=1.2), dict(port='inwards', name='inwards.y')),
        ('v', dict(), dict(port='inwards', name='inwards.v', mask=[True, True, True])),
        ('v', dict(lower_bound=0), dict(port='inwards', name='inwards.v', mask=[True, True, True], lower_bound=0)),
        ('v[::2]', dict(), dict(port='inwards', name='inwards.v', mask=[True, False, True])),
        ('v[:-1]', dict(), dict(port='inwards', name='inwards.v', mask=[True, True, False])),
    ])
    def test_to_dict(self, name, options, expected):
        def get_expected(key, default=None):
            """Set expected[key] to options[key] if `key` is not specified in `expected`.
            Use `default` if all else fails. Returns expected[key]."""
            return expected.setdefault(key, options.get(key, default))

        system = TestUnknown.ASyst('a')
        
        unknown = Unknown(system, name, **options)
        unknown_dict = unknown.to_dict()
        assert unknown_dict["context"] == system.contextual_name
        assert unknown_dict["name"] == expected.get('name', name)
        assert unknown_dict["lower_bound"] == get_expected('lower_bound', -np.inf)
        assert unknown_dict["upper_bound"] == get_expected('upper_bound', np.inf)
        assert unknown_dict["max_abs_step"] == get_expected('max_abs_step', np.inf)
        assert unknown_dict["max_rel_step"] == get_expected('max_rel_step', np.inf)
        np.testing.assert_array_equal(unknown_dict["mask"], expected.get('mask', None))


class TestUnknownIntegration:
    class CustomPort(Port):
        def setup(self):
            self.add_variable('x')
            self.add_variable('y')

    class CustomSystem(System):
        def setup(self):
            self.add_input(TestUnknownIntegration.CustomPort, 'p_in')
            self.add_output(TestUnknownIntegration.CustomPort, 'p_out')

        def compute(self):
            self.p_out.x = self.p_in.x
            self.p_out.y = self.p_in.y


    @pytest.mark.parametrize("inner_unknown", [True, False])
    @pytest.mark.parametrize("y_target, unknown_options, solver_options, expected", [
        (80.0, dict(), dict(), dict(y=80, tol=1e-10, successful=True)),
        (-1e6, dict(lower_bound=-30, max_abs_step=1), dict(), dict(y=-30, tol=0, successful=False)),
        (80.0, dict(upper_bound=10, max_rel_step=0.25), dict(), dict(y=10, tol=0, successful=False)),
        (80.0, dict(max_rel_step=0.05), dict(max_iter=1), dict(y=1.05, tol=0, successful=False)),
        (80.0, dict(max_rel_step=0.05), dict(max_iter=2), dict(y=1.1025, tol=0, successful=False)),
        (80.0, dict(max_abs_step=1), dict(max_iter=1), dict(y=2, tol=0, successful=False)),
        (80.0, dict(max_abs_step=1), dict(max_iter=2), dict(y=3, tol=0, successful=False)),
    ])
    def test_bounds(self, caplog, inner_unknown, y_target, unknown_options, solver_options, expected):
        """Test resolution with specified lower or upper bounds"""
        class CustomSystem(TestUnknownIntegration.CustomSystem):
            def setup(self):
                super().setup()
                self.add_unknown('p_in.y', **unknown_options)

        solver_options.setdefault("max_iter", 200)
        solver_options.setdefault("method", NonLinearMethods.NR)
        equation = f"p_out.y == {y_target}"

        def make_case(SystemClass):
            s = SystemClass('s')
            solver = s.add_driver(NonLinearSolver('solver', **solver_options))
            return s, solver, solver.add_child(RunSingleCase('run'))

        if inner_unknown:  # system with embedded unknown
            s, solver, runner = make_case(CustomSystem)
            runner.offdesign.add_equation(equation)
        else:
            s, solver, runner = make_case(TestUnknownIntegration.CustomSystem)
            runner.design.add_unknown('p_in.y', **unknown_options).add_equation(equation)

        caplog.clear()
        with caplog.at_level(logging.ERROR):
            s.run_drivers()
        assert s.p_in.y == pytest.approx(expected['y'], rel=expected.get('tol', None))

        if expected['successful']:
            assert len(caplog.records) == 0
        else:
            assert len(caplog.records) == 1
            for record in caplog.records:
                assert re.match(
                    r"The solver failed:    -> Not converged \((\d)+(?:\.\d*)(?:[eE][+-]\d+)\) in \d+ iterations"
                    r", \d+ complete, \d+ partial Jacobian and \d+ Broyden evaluation\(s\)",
                    record.message)
