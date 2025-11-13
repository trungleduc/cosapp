import sys
from pathlib import Path

import pytest
import numpy as np
from cosapp.base import System


@pytest.fixture
def test_library():
    library_path = Path(__file__).parent / "library" / "systems"

    # Add path to allow System to find the component
    sys.path.append(str(library_path))
    try:
        yield library_path
    finally:
        # Undo path modification
        sys.path.remove(str(library_path))


@pytest.fixture
def test_data():
    return Path(__file__).parent / "data"


class MassFreeFall(System):
    """Free fall of a point mass, with linear friction model"""
    def setup(self):
        self.add_inward("mass", 1.2, desc="Mass")
        self.add_inward("cf", 0.1, desc="Friction coefficient")
        self.add_inward("g", np.r_[0, 0, -9.81], desc="Uniform acceleration field")

        self.add_outward("a", np.zeros(3))

        self.add_transient("v", der="a")
        self.add_transient("x", der="v")

    def compute(self):
        self.a = self.g - np.expand_dims(self.cf / self.mass, axis=-1) * self.v


class FreeFallSolution:
    """Analytical solution of dynamic system `MassFreeFall`"""
    def __init__(self, system, v0, x0=np.zeros(3)):
        mass = system.mass
        if np.isscalar(mass):
            if mass <= 0:
                raise ValueError("Mass must be strictly positive")
        else:
            mass = np.asarray(mass)
            if any(mass <= 0):
                raise ValueError("All masses must be strictly positive")
        self._x0 = np.atleast_2d(x0).T
        self._v0 = np.atleast_2d(v0).T
        self._g = g = np.atleast_2d(system.g).T
        try:
            cf = system.cf
        except AttributeError:
            cf = system.k
        if cf > 0:
            omega = cf / mass
            self._A = g / omega
        else:
            omega = 0.0
            self._A = np.full_like(g, np.inf)
        self._B = self._v0 - self._A
        self.__omega = omega

    @property
    def omega(self) -> float:
        return self.__omega

    def a(self, t):
        """Acceleration at time t"""
        v = self.__v(t)
        a = self._g - v * self.__omega
        return np.squeeze(a.T)

    def v(self, t):
        """Velocity at time t"""
        v = self.__v(t)
        return np.squeeze(v.T)

    def x(self, t):
        """Position at time t"""
        t = np.asarray(t)
        omega = self.__omega
        wt = omega * t
        x0 = self._x0
        v0 = self._v0
        with np.errstate(invalid="ignore"):
            x = np.where(
                wt < 1e-7,  # asymptotic expansion, to avoid exp overflow
                x0 + v0 * t + (0.5 * t) * (self._g * t - wt * v0) * (1 - wt / 3 * (1 - 0.25 * wt)),
                x0 + self._A * t + self._B / omega * (1 - np.exp(-wt)),
            )
        return np.squeeze(x.T)

    def __v(self, t):
        t = np.asarray(t)
        wt = self.__omega * t
        v0 = self._v0
        with np.errstate(invalid="ignore"):
            return np.where(
                wt < 1e-7,  # asymptotic expansion, to avoid exp overflow
                v0 + (self._g * t - v0 * wt) * (1 - wt * (0.5 - wt / 6)),
                self._A + self._B * np.exp(-wt),
            )
