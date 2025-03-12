import numpy as np


class PointMassSolution:
    """Analytical solution of dynamic system PointMass"""
    def __init__(self, system, v0, x0=np.zeros(3)):
        if system.mass <= 0:
            raise ValueError("Mass must be strictly positive")
        self._x0 = np.atleast_2d(x0).T
        self._v0 = np.atleast_2d(v0).T
        self._g = g = np.atleast_2d(system.g).T
        try:
            cf = system.cf
        except AttributeError:
            cf = system.k
        if cf > 0:
            omega = cf / system.mass
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
        with np.errstate(invalid='ignore'):
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
        with np.errstate(invalid='ignore'):
            return np.where(
                wt < 1e-7,  # asymptotic expansion, to avoid exp overflow
                v0 + (self._g * t - v0 * wt) * (1 - wt * (0.5 - wt / 6)),
                self._A + self._B * np.exp(-wt),
            )


class CoupledTanksSolution:
    """Analytical solution of dynamic system CoupledTanks"""
    def __init__(self, system, init):
        K = 9.81 * system.tank1.rho * system.pipe.k
        a = system.tank1.area / system.tank2.area
        self.__area_ratio = a
        self.__tau = system.tank1.area / ((1 + a) * K)
        self.initial_heights = init

    @property
    def characteristic_time(self):
        return self.__tau

    def __call__(self, t):
        tau = self.__tau
        a = self.__area_ratio
        h1_0, h2_0 = self.initial_heights
        dh = (h1_0 - h2_0) * np.exp(-t / tau)
        h1 = (a * h1_0 + h2_0 + dh) / (1 + a)
        h2 = h1 - dh
        return (h1, h2)


class HarmonicOscillatorSolution:
    def __init__(self, system, init=(0, 0)):
        K = system.K
        c = system.c
        m = system.mass
        L = system.length
        w0 = np.sqrt(K / m)
        self.__damping = dc = 0.5 * c / np.sqrt(m * K)
        a = w0 * dc
        x0, v0 = init
        x0 -= L
        if self.over_damped:
            wd = w0 * np.sqrt(dc**2 - 1)
            A, B = 0.5 * (v0 + (a + wd) * x0) / wd, 0.5 * (v0 + (a - wd) * x0) / wd
            self.__x = lambda t: L + (A * np.exp(-(a - wd) * t) - B * np.exp(-(a + wd) * t))
        else:
            wd = w0 * np.sqrt(1 - dc**2)
            A, B = (v0 + a * x0) / wd, x0
            self.__x = lambda t: L + np.exp(-a * t) * (A * np.sin(wd * t) + B * np.cos(wd * t))
        
    def x(self, t):
        return self.__x(t)

    @property
    def damping(self):
        """Damping coefficient"""
        return self.__damping
    
    @property
    def over_damped(self):
        return self.__damping > 1