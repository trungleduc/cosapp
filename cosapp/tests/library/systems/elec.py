"""Systems from tutorials used in regression tests"""

from __future__ import annotations
from cosapp.base import System, Port
import abc


class Voltage(Port):
    def setup(self):
        self.add_variable("V", unit="V")

        
class Current(Port):
    def setup(self):
        self.add_variable("I", unit="A")


class Dipole(System):
    """Simple dipole model of the kind: {V_in, V_out} -> I"""
    def setup(self):
        self.add_input(Voltage, "V_in")
        self.add_input(Voltage, "V_out")
        self.add_output(Current, "I")
        self.add_outward("deltaV", 0.0)
    
    def compute(self):
        self.deltaV = self.V_in.V - self.V_out.V
        self.I.I = self.compute_I()

    @abc.abstractmethod
    def compute_I(self) -> float:
        pass


class Resistor(Dipole):
    """Linear resistor model: {V_in, V_out} -> I"""
    def setup(self, R=100.):
        super().setup()
        self.add_inward("R", float(R), unit="ohm", desc="Resistance")

    def compute_I(self) -> float:
        return self.deltaV / self.R


class Capacitor(Dipole):
    """Simple capacitor model: {V_in, V_out} -> I"""
    def setup(self, C=0.002):
        super().setup()
        self.add_inward("C", float(C), desc="Capacity")
        # self.add_rate("dUdt", source="deltaV")
        self.add_inward("dUdt", 0.0)
        self.add_transient("U", der="dUdt")
        self.add_unknown("dUdt").add_equation("U == deltaV")

    def compute_I(self) -> float:
        return self.C * self.dUdt


class Node(System):
    """Electric node model from Kirchhoff laws"""

    def setup(self, n_in=1, n_out=1):
        self.add_property("n_in", int(n_in))
        self.add_property("n_out", int(n_out))

        if min(self.n_in, self.n_out) < 1:
            raise ValueError("Node needs at least one incoming and one outgoing current")

        self.add_property(
            "incoming", tuple(
                self.add_input(Current, f"I_in{i}")
                for i in range(self.n_in)
            )
        )
        self.add_property(
            "outgoing", tuple(
                self.add_input(Current, f"I_out{i}")
                for i in range(self.n_out)
            )
        )
        
        self.add_inward("V", 0.0, unit="V")
        self.add_outward("sum_I_in", 0.0, desc="Total incoming current")
        self.add_outward("sum_I_out", 0.0, desc="Total outgoing current")
        
        # Off-design problem
        self.add_unknown("V").add_equation(
            "sum_I_in == sum_I_out",
            name="Current balance",
        )

    def compute(self):
        self.sum_I_in = sum(current.I for current in self.incoming)
        self.sum_I_out = sum(current.I for current in self.outgoing)

    @classmethod
    def make(cls, parent: System, name: str, incoming: list[Dipole]=[], outgoing: list[Dipole]=[], pulling=None) -> Node:
        """Factory method making appropriate connections with parent system"""
        node = cls(name, n_in=max(len(incoming), 1), n_out=max(len(outgoing), 1))
        parent.add_child(node, pulling=pulling)
        
        for dipole, current_port in zip(incoming, node.incoming):
            parent.connect(dipole.V_out, node.inwards, "V")
            parent.connect(dipole.I, current_port)
        
        for dipole, current_port in zip(outgoing, node.outgoing):
            parent.connect(dipole.V_in, node.inwards, "V")
            parent.connect(dipole.I, current_port)

        return node


class Source(System):
    def setup(self, I=0.1):
        self.add_inward("I", I, unit="A")
        self.add_output(Current, "I_out", {"I": I})
    
    def compute(self):
        self.I_out.I = self.I


class Ground(System):
    def setup(self, V=0.0):
        self.add_inward("V", V, unit="V")
        self.add_output(Voltage, "V_out", {"V": V})
    
    def compute(self):
        self.V_out.V = self.V
