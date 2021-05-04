import pytest
import numpy as np

from cosapp.ports import Port
from cosapp.systems import System
from cosapp.core.numerics.residues import DeferredResidue, Residue


class UvPort(Port):
    def setup(self):
        self.add_variable('u', 1.0)
        self.add_variable('v', np.zeros(3))


class LocalSystem(System):
    def setup(self):
        self.add_property('g', 9.81)

        self.add_input(UvPort, 'uv_in')
        self.add_inward('a', 1.)
        self.add_inward('b', np.array([1, 2], dtype=float))

        self.add_output(UvPort, 'uv_out')
        self.add_outward('x', 0.0)
        self.add_outward('y', 0.0)


@pytest.fixture(scope='function')
def context():
    context = LocalSystem('context')
    context.add_child(LocalSystem('sub'))
    return context


@pytest.mark.parametrize("settings, expected", [
    (
        dict(target='a'),
        dict(variables={'a'}),
    ),
    (
        dict(target='x'),
        dict(variables={'x'}),
    ),
    (
        dict(target='x', reference=0.1),
        dict(variables={'x'}),
    ),
    (
        dict(target='sub.x'),
        dict(variables={'sub.x'}),
    ),
    (
        dict(target='a * x'),
        dict(variables={'a', 'x'}),
    ),
    (
        dict(target='a * g'),
        dict(variables={'a'}),
    ),
    (
        dict(target='a * g + uv_in.u - uv_out.v**2'),
        dict(variables={'a', 'uv_in.u', 'uv_out.v'}),
    ),
])
def test_DeferredResidue__init__(context, settings, expected):
    error = expected.get('error', None)

    if error is None:
        deferred = DeferredResidue(context, **settings)
        assert deferred.context is context
        assert deferred.target == expected.get('target', settings['target'])
        assert deferred.reference == settings.get('reference', 1)
        assert deferred.variables == expected['variables']

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            DeferredResidue(context, **settings)


def test_DeferredResidue_make_residue(context):
    deferred = DeferredResidue(context, target='y')
    context.y = 3.14159
    residue = deferred.make_residue()
    assert isinstance(residue, Residue)
    assert residue.context is deferred.context
    assert residue.equation == "y == 3.14159"
    assert residue.reference == deferred.reference
    assert residue.value == 0

    deferred = DeferredResidue(context, target='a * y + g')
    context.a = 0.5
    context.y = 4.0
    residue = deferred.make_residue()
    assert isinstance(residue, Residue)
    assert residue.context is deferred.context
    assert residue.equation == "a * y + g == 11.81"
    assert residue.reference == deferred.reference
    assert residue.value == 0


@pytest.mark.parametrize("options, expected", [
    (dict(), dict()),
    (dict(reference=0.1), dict()),
    (dict(reference='norm'), dict(reference=1e-2)),
    (dict(reference=-2.5), dict(error=ValueError)),
    (dict(reference='nonsense'), dict(error=ValueError)),
])
def test_DeferredResidue_make_residue_with_ref(context, options, expected):
    error = expected.get('error', None)
    deferred = DeferredResidue(context, target='y', reference=1.0)

    if error is None:
        context.y = 3.14159e-2
        residue = deferred.make_residue(**options)
        assert isinstance(residue, Residue)
        assert residue.context is deferred.context
        assert residue.equation == "y == 0.0314159"
        # Expected reference: try `expected`, then `options`, then `deferred`
        expected_ref = expected.get('reference', options.get('reference', deferred.reference))
        assert residue.reference == expected_ref
        assert residue.value == 0

    else:
        with pytest.raises(error, match=expected.get('match', None)):
            deferred.make_residue(**options)
