import pytest

from cosapp.ports import Port, PortType
from cosapp.tests.library.systems import Multiply1, Multiply2, MultiplySystem, ComplexTurbofan


class TestMultiplyFamily:
    """Test SystemFamily through derived class MultiplyFamily"""
    def __init__(self):
        self.s = MultiplySystem("head")
        self.s1 = Multiply1("mult")

    def test_modelings(self):
        assert len(self.s1.modelings) == 3

    def test_bestratio(self):
        assert self.s1.modelings.best_fidelity_to_cost_ratio() == "Multiply1"

    def test_highestfidelity(self):
        assert self.s1.modelings.highest_fidelity() == "Multiply3"

    def test_lowestfidelity(self):
        assert self.s1.modelings.lowest_fidelity() == "Multiply1"

    def test_highestcost(self):
        assert self.s1.modelings.highest_cost() == "Multiply3"

    def test_lowestcost(self):
        assert self.s1.modelings.lowest_cost() == "Multiply1"

    def test_delete(self):
        assert not self.s1.modelings.exists("MyModeling")
        assert self.s1.modelings.exists("Multiply3")
        self.s1.modelings.delete("Multiply3")
        assert not self.s1.modelings.exists("Multiply3")

    def test_possible_conversions(self):
        assert self.s1.possible_conversions() == [
            "Multiply1_to_Multiply2",
            "Multiply1_to_Multiply3",
        ]

    def test_plug_same(self):
        s = self.s
        s.mult1.convert_to("Multiply2")

        connectors = s.connectors()
        assert set(connectors) == {
            "p_in -> mult1.p_in",
            "mult1.p_out -> p_out",
        }
        connector = connectors["p_in -> mult1.p_in"]
        assert connector.source.owner is s
        assert connector.source is s.p_in

        class Fake(Port):
            def setup(self):
                self.add_variable("none")

        s.mult1.outputs["fake_port"] = Fake("fake_port", PortType.OUT, {"none": 1.})
        s.mult1.outputs["fake_port"].owner = s.mult1
        s.mult1.convert_to("Multiply3")

        connectors = s.connectors()
        assert set(connectors) == {
            "p_in -> mult1.p_in",
            "mult1.p_out -> p_out",
        }
        connector = connectors["p_in -> mult1.p_in"]
        assert connector.source.owner is s
        assert connector.source is s.p_in

    def test_convert_to(self):
        s = self.s
        s1 = Multiply1("mult")

        assert type(s.mult1).__name__ == "Multiply1"

        with pytest.raises(ValueError, match='same classes'):
            s.mult1.convert_to("Multiply1")

        with pytest.raises(ValueError, match='method does not exist'):
            s.mult1.convert_to("Multipl")

        with pytest.raises(TypeError, match="system is not part of a family"):
            s.convert_to("Example")

        with pytest.raises(ValueError, match='parent system is immutable'):
            s1.convert_to("Multiply2")

        with pytest.raises(TypeError, match="argument 'to_type' should be str"):
            s.mult1.convert_to(Multiply2)

    def test_conversions(self):
        s = self.s
        s.p_in.x = 10
        s.mult1.inwards.K1 = 25

        s.run_once()
        assert s.mult1.p_out.x == 250

        res = s.mult1.convert_to("Multiply2")
        assert res is None
        assert type(s.mult1).__name__ == "Multiply2"
        assert s.mult1.inwards.K1 == 25
        assert s.mult1.inwards.K2 == 5
        s.run_once()
        assert s.mult1.p_out.x == 250

        res = s.mult1.convert_to("Multiply1")
        assert res is None
        assert type(s.mult1).__name__ == "Multiply1"
        assert s.mult1.inwards.K1 == 25
        s.run_once()
        assert s.mult1.p_out.x == 250

        s.mult1.inwards.K1 = 125
        res = s.mult1.convert_to("Multiply3")
        assert res is None
        assert type(s.mult1).__name__ == "Multiply3"
        assert s.mult1.inwards.K1 == 125
        assert s.mult1.inwards.K2 != pytest.approx(5, abs=1e-6)
        assert s.mult1.inwards.K3 != pytest.approx(5, abs=1e-6)
        s.mult1.K1 = s.mult1.K2 = s.mult1.K3 = 5
        s.run_once()
        assert s.mult1.p_out.x == pytest.approx(1250, rel=1e-12)

        res = s.mult1.convert_to("Multiply1")
        assert res is None
        assert type(s.mult1).__name__ == "Multiply1"
        assert s.mult1.inwards.K1 == pytest.approx(125, rel=1e-12)
        s.run_once()
        assert s.mult1.p_out.x == pytest.approx(1250, rel=1e-12)


def test_SystemFamily_convert_to():
    s = ComplexTurbofan("MyComplexTurbofan")

    s.fanC.ductC.convert_to("Duct")

    assert "fl_in" in s.fanC.ductC.inputs
    assert "fl_out" in s.fanC.ductC.outputs
    assert s.fanC.ductC.parent is s.fanC

    connectors = s.fanC.connectors()
    assert set(connectors) == {
        'inwards -> fan.inwards',
        'fl_in -> ductC.fl_in',
        'mech_in -> fan.mech_in',
        'ductC.fl_out -> fan.fl_in',
        'fan.fl_out -> fl_out',
    }
    connector = connectors["fl_in -> ductC.fl_in"]
    assert connector.source is s.fanC.fl_in
    assert connector.sink is s.fanC.ductC.fl_in
    
    connector = connectors["ductC.fl_out -> fan.fl_in"]
    assert connector.source is s.fanC.ductC.fl_out
    assert connector.sink is s.fanC.fan.fl_in

    with pytest.raises(TypeError):
        s.fanC.ductC.convert_to(1.0)

    with pytest.raises(TypeError):
        s.fanC.ductC.convert_to(list())

    with pytest.raises(ValueError):
        s.fanC.ductC.convert_to("foobar")

    with pytest.raises(ValueError):
        s.fanC.ductC.convert_to("")
