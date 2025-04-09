import pytest
import numpy as np

from cosapp.drivers.time.utils import TwoPointCubicPolynomial


@pytest.mark.parametrize("xs", [[1, 2], [0, 0.5], [-0.2, 0.1]])
@pytest.mark.parametrize("ys", [[0.7, -0.2], [0, 1]])
@pytest.mark.parametrize("dy", [[-0.5, 0.1], [0, 1], [0, 0]])
def test_TwoPointCubicPolynomial_scalars(xs, ys, dy):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    dy = np.asarray(dy)
    p = TwoPointCubicPolynomial(xs, ys, dy)
    dp = p.deriv()

    assert p.degree() == 3
    assert p(xs) == pytest.approx(ys, abs=1e-14)
    assert dp(xs) == pytest.approx(dy, abs=1e-14)


@pytest.mark.parametrize("xs", [[1, 2], [0, 0.5], [-0.2, 0.1]])
@pytest.mark.parametrize("ys, dy", [
    ([0.7, -0.2], [-0.5, 0.1]),
    ([0, 1], [0, 1]),
    (   # 3D vector data
        [[-0.7, 0.1, 0.4], [1., -0.3, 0.2]],
        [[0.22, 1.4, 0.0], [0., 0.9, -0.6]],
    ),
    (   # (1, 3) array data
        [[[-0.7, 0.1, 0.4]], [[1., -0.3, 0.2]]],
        [[[0.22, 1.4, 0.0]], [[0., 0.9, -0.6]]],
    ),
    (   # (2, 3) array data
        [[[-0.7, 0.1, 0.4], [1., -0.3, 0.2]], [[0.5, -0.1, 1.4], [0.0, 0.7, 0.3]]],
        [[[0.22, 1.4, 0.0], [0., 0.9, -0.6]], [[1.23, 0.4, 0.7], [-0.8, 1.5, 0.0]]],
    ),
])
def test_TwoPointCubicPolynomial_arrays(xs, ys, dy):
    p = TwoPointCubicPolynomial(xs, ys, dy)
    dp = p.deriv()

    np.testing.assert_allclose(p(xs[0]), ys[0], rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(p(xs[1]), ys[1], rtol=1e-15, atol=1e-15)

    np.testing.assert_allclose(dp(xs[0]), dy[0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(dp(xs[1]), dy[1], rtol=1e-14, atol=1e-14)


def test_numpy_polynomial():
    p = np.polynomial.Polynomial([0, 1], [1, 2], [0, 1])

    assert p(0) == pytest.approx(-1, abs=1e-14)
    assert p(2) == pytest.approx(+1, abs=1e-14)
