import numpy as np
from cosapp.base import System


class SellarDiscipline1(System):
    """Component containing Discipline 1.
    """
    def setup(self):
        self.add_inward("z", np.zeros(2))
        self.add_inward("x", 0.0)
        self.add_inward("y2", 0.0)

        self.add_outward("y1", 0.0)

    def compute(self):
        """Evaluates equation
        y1 = z1**2 + z2 + x - 0.2 * y2
        """
        self.y1 = self.z[0]**2 + self.z[1] + self.x - 0.2 * self.y2


class SellarDiscipline2(System):
    """Component containing Discipline 2.
    """
    def setup(self):
        self.add_inward("z", np.zeros(2))
        self.add_inward("y1", 0.0)

        self.add_outward("y2", 0.0)

    def compute(self):
        """Evaluates equation
        y2 = sqrt(|y1|) + z1 + z2
        """
        self.y2 = np.sqrt(abs(self.y1)) + self.z[0] + self.z[1]


class Sellar(System):
    """System modeling the Sellar case.
    """
    def setup(self):
        d1 = self.add_child(SellarDiscipline1("d1"), pulling=["x", "z", "y1"])
        d2 = self.add_child(SellarDiscipline2("d2"), pulling=["z", "y2"])

        # Couple sub-systems d1 and d2:
        self.connect(d1, d2, ["y1", "y2"])
