import pytest
import numpy as np
from numpy.polynomial import Polynomial

from cosapp.drivers.time.bdf import BdfWeights


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_BdfWeights_1_iso(dt):
    """Test BDF-3 with regular steps."""
    bdf = BdfWeights(1)

    assert bdf.order == 1
    assert np.array_equal(bdf.steps, np.zeros(1))
    assert np.array_equal(bdf.weights, np.zeros(2))

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [dt])
    assert dt * bdf.weights == pytest.approx([-1.0, 1.0], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [dt])
    assert dt * bdf.weights == pytest.approx([-1.0, 1.0], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_BdfWeights_2_iso(dt):
    """Test BDF-3 with regular steps."""
    bdf = BdfWeights(2)

    assert bdf.order == 2
    assert np.array_equal(bdf.steps, np.zeros(2))
    assert np.array_equal(bdf.weights, np.zeros(3))

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [0.0, dt])
    assert dt * bdf.weights == pytest.approx([0.0, -1.0, 1.0], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [dt, dt])
    assert dt * bdf.weights == pytest.approx([0.5, -2.0, 1.5], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [dt, dt])
    assert dt * bdf.weights == pytest.approx([0.5, -2.0, 1.5], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_BdfWeights_3_iso(dt):
    """Test BDF-3 with regular steps."""
    bdf = BdfWeights(3)

    assert bdf.order == 3
    assert np.array_equal(bdf.steps, np.zeros(3))
    assert np.array_equal(bdf.weights, np.zeros(4))

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [0.0, 0.0, dt])
    assert dt * bdf.weights == pytest.approx([0.0, 0.0, -1.0, 1.0], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [0.0, dt, dt])
    assert dt * bdf.weights == pytest.approx([0.0, 0.5, -2.0, 1.5], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [dt, dt, dt])
    assert dt * bdf.weights == pytest.approx([-1/3, 1.5, -3.0, 11/6], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [dt, dt, dt])
    assert dt * bdf.weights == pytest.approx([-1/3, 1.5, -3.0, 11/6], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)


@pytest.mark.parametrize("dt", [0.01, 0.1, 1.0])
def test_BdfWeights_4_iso(dt):
    """Test BDF-4 with regular steps."""
    bdf = BdfWeights(4)

    assert bdf.order == 4
    assert np.array_equal(bdf.steps, np.zeros(4))
    assert np.array_equal(bdf.weights, np.zeros(5))

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [0.0, 0.0, 0.0, dt])
    assert dt * bdf.weights == pytest.approx([0.0, 0.0, 0.0, -1.0, 1.0], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [0.0, 0.0, dt, dt])
    assert dt * bdf.weights == pytest.approx([0.0, 0.0, 0.5, -2.0, 1.5], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [0.0, dt, dt, dt])
    assert dt * bdf.weights == pytest.approx([0.0, -1/3, 1.5, -3.0, 11/6], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [dt, dt, dt, dt])
    assert dt * bdf.weights == pytest.approx([0.25, -4/3, 3.0, -4.0, 25/12], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)

    bdf.push_step(dt)
    assert np.array_equal(bdf.steps, [dt, dt, dt, dt])
    assert dt * bdf.weights == pytest.approx([0.25, -4/3, 3.0, -4.0, 25/12], rel=1e-14)
    assert dt * bdf.weights.sum() == pytest.approx(0.0, abs=2.5e-15)


@pytest.mark.parametrize("x", [
    # Three evaluation points defining two steps
    np.r_[1.0, 1.2, 1.7],
    np.r_[-0.5, -0.2, -0.1],
    np.r_[-0.5, -0.2, -0.1] + 10,
    np.r_[-0.5, -0.2, -0.1] * 10,
])
@pytest.mark.parametrize("coefs", [
    # Second-order polynomial coefficients
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [2.0, -1.5, 0.5],
    [0.0, 1.5, -2.5],
])
def test_BdfWeights_2_polynomial(coefs, x):
    """Test BDF-2 weights on a second-order polynomial (end-point derivative should be exact)."""
    x = np.array(x)  # force copy to avoid side effects
    p = Polynomial(coefs)
    dp = p.deriv()

    bdf = BdfWeights(2)

    for dx in np.diff(x):
        bdf.push_step(dx)

    assert bdf.order == 2
    assert p.degree() == 2
    assert np.array_equal(bdf.steps, np.diff(x))
    assert np.dot(bdf.weights, p(x)) == pytest.approx(dp(x[-1]), rel=1e-13)

    # Update last step and recheck values
    x[-1] += x[-1] - x[-2]
    bdf.replace_last_step(x[-1] - x[-2])
    assert np.array_equal(bdf.steps, np.diff(x))
    assert np.dot(bdf.weights, p(x)) == pytest.approx(dp(x[-1]))


@pytest.mark.parametrize("x", [
    # Four evaluation points defining three steps
    np.r_[1.0, 1.2, 1.3, 1.35],
    np.r_[-0.5, -0.2, -0.1, 0.5],
    np.r_[-0.5, -0.2, -0.1, 0.5] + 10,
    np.r_[-0.5, -0.2, -0.1, 0.5] * 10,
])
@pytest.mark.parametrize("coefs", [
    # third-order polynomial coefficients
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 2.0, -1.5, 0.5],
    [0.0, -2.0, 1.5, 2.5],
])
def test_BdfWeights_3_polynomial(coefs, x):
    """Test BDF-3 weights on a third-order polynomial (end-point derivative should be exact)."""
    x = np.array(x)  # force copy to avoid side effects
    p = Polynomial(coefs)
    dp = p.deriv()

    bdf = BdfWeights(3)

    for dx in np.diff(x):
        bdf.push_step(dx)

    assert bdf.order == 3
    assert p.degree() == 3
    assert np.array_equal(bdf.steps, np.diff(x))
    assert np.dot(bdf.weights, p(x)) == pytest.approx(dp(x[-1]), rel=1e-13)

    # Update last step and recheck values
    x[-1] += x[-1] - x[-2]
    bdf.replace_last_step(x[-1] - x[-2])
    assert np.array_equal(bdf.steps, np.diff(x))
    assert np.dot(bdf.weights, p(x)) == pytest.approx(dp(x[-1]))


@pytest.mark.parametrize("x", [
    # Five evaluation points defining four steps
    np.r_[1.0, 1.2, 1.3, 1.35, 1.5],
    np.r_[-0.5, -0.2, -0.1, 0.5, 0.51],
    np.r_[-0.5, -0.2, -0.1, 0.5, 0.51] + 10,
    np.r_[-0.5, -0.2, -0.1, 0.5, 0.51] * 10,
])
@pytest.mark.parametrize("coefs", [
    # fourth-order polynomial coefficients
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 2.0, -1.5, 0.5],
    [0.0, 2.0, -2.0, 1.5, 2.5],
])
def test_BdfWeights_4_polynomial(coefs, x):
    """Test BDF-4 weights on a fourth-order polynomial (end-point derivative should be exact)."""
    x = np.array(x)  # force copy to avoid side effects
    p = Polynomial(coefs)
    dp = p.deriv()

    bdf = BdfWeights(4)

    for dx in np.diff(x):
        bdf.push_step(dx)

    assert bdf.order == 4
    assert p.degree() == 4
    assert np.array_equal(bdf.steps, np.diff(x))
    assert np.dot(bdf.weights, p(x)) == pytest.approx(dp(x[-1]), rel=1e-13)

    # Update last step and recheck values
    x[-1] += x[-1] - x[-2]
    bdf.replace_last_step(x[-1] - x[-2])
    assert np.array_equal(bdf.steps, np.diff(x))
    assert np.dot(bdf.weights, p(x)) == pytest.approx(dp(x[-1]))


@pytest.mark.parametrize("order", range(1, 5))
def test_BdfWeights_push_step(order):
    """Test `BdfWeights.push_step`"""
    bdf = BdfWeights(order)

    assert bdf.order == order

    dx = 0.125
    for dx in np.linspace(0.1, 0.2, 11):
        bdf.push_step(dx)
        bdf.steps[-1] == dx

    with pytest.raises(ValueError):
        bdf.push_step(0.0)

    with pytest.raises(ValueError):
        bdf.push_step(-1.0)


@pytest.mark.parametrize("order, valid", [
    (1, True),
    (2, True),
    (3, True),
    (4, True),
    (5, False),
    (0, False),
    (-1, False),
])
def test_BdfWeights_order(order, valid):

    if valid:
        bdf = BdfWeights(order)
        assert bdf.order == order
        assert np.array_equal(bdf.steps, np.zeros(order))
        assert np.array_equal(bdf.weights, np.zeros(order + 1))
    
    else:
        with pytest.raises(ValueError):
            BdfWeights(order)
