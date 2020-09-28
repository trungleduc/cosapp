from typing import Union

import pytest

from cosapp.core.numerics.residues import Residue
from cosapp.ports.port import ExtensiblePort, Port
from cosapp.systems.system import IterativeConnector, System
from cosapp.tests.library.ports import XPort
from cosapp.tests.library.systems import ComplexDuct


def test_IterativeConnector___init___():
    s = ComplexDuct("MyComplexDuct")
    connector = s.connectors["bleed_fl2_out_to_merger_fl2_in"]
    connection = IterativeConnector(connector)

    assert isinstance(connection, IterativeConnector)
    assert connection._sink is s.merger.fl2_in
    assert connection._source is s.bleed.fl2_out
    assert connection._mapping is connector.variable_mapping
    assert len(connection.inputs) == 3
    assert len(connection.inputs[System.INWARDS]) == 0
    assert len(connection.inputs[IterativeConnector.GUESS]) == len(s.merger.fl2_in)
    assert isinstance(
        connection.inputs[IterativeConnector.GUESS], type(s.merger.fl2_in)
    )
    assert len(connection.inputs[IterativeConnector.RESULT]) == len(s.bleed.fl2_out)
    assert isinstance(
        connection.inputs[IterativeConnector.RESULT], type(s.bleed.fl2_out)
    )

    for r in ("iterative_Tt", "iterative_Pt", "iterative_W"):
        assert isinstance(connection.residues[r], Residue)

    class S(System):
        def setup(self):
            self.add_inward("a_in")
            self.add_inward("b_in")
            self.add_input(XPort, "entry")
            self.add_output(XPort, "exit")
            self.add_outward("a_out")
            self.add_outward("b_out")

        def compute(self):
            self.exit.x = self.entry.x * self.a_in + self.b_in
            self.a_out = self.entry.x * self.a_in
            self.b_out = self.b_in / self.a_in

    s = System("group")
    a = s.add_child(S("a"))
    b = s.add_child(S("b"))
    s.connect(a.inwards, b.outwards, {"a_in": "a_out"})
    s.connect(a.exit, b.entry)
    connector = s.connectors["b_outwards_to_a_inwards"]
    connection = IterativeConnector(connector)

    assert len(connection.inputs) == 3
    assert len(connection.inputs[System.INWARDS]) == 0
    assert isinstance(connection.inputs[System.INWARDS], ExtensiblePort)
    assert len(connection.inputs[IterativeConnector.GUESS]) == 1
    assert isinstance(connection.inputs[IterativeConnector.GUESS], ExtensiblePort)
    assert "a_in" in connection.inputs[IterativeConnector.GUESS]
    assert len(connection.inputs[IterativeConnector.RESULT]) == 1
    assert isinstance(connection.inputs[IterativeConnector.RESULT], ExtensiblePort)
    assert "a_out" in connection.inputs[IterativeConnector.RESULT]


def test_IterativeConnector_compute():
    s = ComplexDuct("MyComplexDuct")
    s.open_loops()

    s["merger_fl2_in.W"] = 5.0
    s["merger_fl2_in.Pt"] = 1000.0

    s.fl_in.W = 95.0
    s.fl_in.Pt = 1020.408163265
    s.run_once()

    assert s.merger.fl2_in.W == pytest.approx(5.0, abs=1e-10)
    assert s.duct.fl_in.W == pytest.approx(100.0, abs=1e-10)
    assert s.bleed.fl_in.W == pytest.approx(100.0, abs=1e-10)
    assert s.bleed.fl1_out.W == pytest.approx(99.0, abs=1e-10)
    assert s.bleed.fl2_out.W == pytest.approx(1.0, abs=1e-10)

    assert s.duct.inwards.cst_loss == pytest.approx(0.98, abs=1e-10)
    assert s.duct.fl_out.Pt == pytest.approx(1000.0, abs=1e-8)

    residue = s.bleed_fl2_out_to_merger_fl2_in.residues["iterative_W"]
    assert residue.value == pytest.approx(
        Residue.evaluate_residue(5.0, 1.0, residue.reference), abs=1e-10
    )
    residue = s.bleed_fl2_out_to_merger_fl2_in.residues["iterative_Pt"]
    assert residue.value == pytest.approx(0.0, abs=5e-10)

    s["merger_fl2_in.W"] = 1.0
    s["merger_fl2_in.Pt"] = 1000.1
    s.run_once()
    residue = s.bleed_fl2_out_to_merger_fl2_in.residues["iterative_W"]
    assert residue.value == pytest.approx(0.04, abs=1e-10)
    residue = s.bleed_fl2_out_to_merger_fl2_in.residues["iterative_Pt"]
    assert residue.value == pytest.approx(
        Residue.evaluate_residue(
            s.merger.fl2_in.Pt, s.bleed.fl2_out.Pt, residue.reference
        ),
        abs=1e-5,
    )


def test_IterativeConnector_get_connection():
    s = ComplexDuct("MyComplexDuct")
    connector = s.connectors["bleed_fl2_out_to_merger_fl2_in"]
    connection = IterativeConnector(connector)

    sink, source, mapping = connection.get_connection()
    assert sink is connector.sink
    assert source is connector.source
    assert mapping == connector.variable_mapping
