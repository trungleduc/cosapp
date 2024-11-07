import pytest
import numpy as np

from cosapp.drivers.time.utils import TwoPointCubicInterpolator


@pytest.mark.parametrize("xs", [[1, 2], [0, 0.5], [-0.2, 0.1]])
@pytest.mark.parametrize("ys, dy", [
    ([0.7, -0.2], [-0.5, 0.1]),
    ([0, 1], [0, 1]),
    (   # 3D vector data
        [[-0.7, 0.1, 0.4], [1, -0.3, 0.2]],
        [[0.22, 1.4, 0.0], [0, 0.9, -0.6]],
    ),
    (   # (1, 3) array data
        [[[-0.7, 0.1, 0.4]], [[1, -0.3, 0.2]]],
        [[[0.22, 1.4, 0.0]], [[0, 0.9, -0.6]]],
    ),
    (   # (2, 3) array data
        [[[-0.7, 0.1, 0.4], [1, -0.3, 0.2]], [[0.5, -0.1, 1.4], [0.0, 0.7, 0.3]]],
        [[[0.22, 1.4, 0.0], [0, 0.9, -0.6]], [[1.23, 0.4, 0.7], [-0.8, 1.5, 0.0]]],
    ),
])
def test_TwoPointCubicInterpolator(xs, ys, dy):
    f = TwoPointCubicInterpolator(xs, ys, dy)

    np.testing.assert_allclose(f(xs[0]), ys[0], rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(f(xs[1]), ys[1], rtol=1e-15, atol=1e-15)
