import os

import numpy as np
import pandas as pd
import pytest

from cosapp.ports import Port
from cosapp.systems import System
from cosapp.systems.metamodels import MetaSystem
from cosapp.systems.surrogate_models.kriging import FloatKrigingSurrogate
from cosapp.systems.surrogate_models.nearest_neighbor import NearestNeighbor
from cosapp.systems.surrogate_models.surrogate_model import SurrogateModel
from cosapp.tests.library.ports import XPort


class OutPort(Port):
    def setup(self):
        self.add_variable("sin")
        self.add_variable("cos")


class YPort(Port):
    def setup(self):
        self.add_variable("y", (0.0, 0.0))


class MetaA(MetaSystem):
    def setup(self):
        self.add_input(XPort, "in_")
        self.add_output(OutPort, "out")


def test_MetaSystem__initialize(test_data):
    x = np.linspace(0, 10, 20)
    data = pd.DataFrame({"x": x, "sin": np.sin(x), "cos": np.cos(x)})
    my_meta = MetaA(
        "metaA", data, mapping={"x": "in_.x", "sin": "out.sin", "cos": "out.cos"}
    )

    assert len(my_meta._input_names) == 1
    assert "in_.x" in my_meta._input_names
    assert my_meta._trained == False
    assert isinstance(my_meta.models, dict)
    assert len(my_meta.models) == 2
    assert issubclass(my_meta._default_model, SurrogateModel)
    assert isinstance(my_meta.inputs[System.INWARDS]["training_data"], pd.DataFrame)

    with pytest.raises(FileNotFoundError):
        my_meta = MetaA("metaA", data=str(test_data / "export_do.csv"))

    with pytest.raises(TypeError):
        my_meta = MetaA("metaA", data=1.0)

    my_meta = MetaA("metaA", data=str(test_data / "export_doe.csv"))
    my_meta = MetaA("metaA", data=str(test_data / "export_doe.cs"))
    try:
        import xlrd
    except ImportError:
        pass
    else:
        my_meta = MetaA("metaA", data=str(test_data / "export_doe.xlsx"))
    my_meta = MetaA("metaA", data=str(test_data / "export_doe.json"))

    assert "K1" in my_meta.training_data.columns
    assert "myK" not in my_meta.training_data.columns

    my_meta = MetaA(
        "metaA", data=str(test_data / "export_doe.csv"), mapping={"K1": "myK"}
    )

    assert "K1" not in my_meta.training_data.columns
    assert "myK" in my_meta.training_data.columns

    my_meta = MetaA("metaA", data=str(test_data / "default_dataframe.json"))
    assert "K1" in my_meta.training_data.columns
    assert "myK" not in my_meta.training_data.columns


def test_MetaSystem__train():
    x = np.linspace(0, 10, 20)
    data = pd.DataFrame({"x": x, "sin": np.sin(x), "cos": np.cos(x)})
    my_meta = MetaA(
        "metaA", data, mapping={"x": "in_.x", "sin": "out.sin", "cos": "out.cos"}
    )

    my_meta._train()
    assert len(my_meta.models) == 2
    assert isinstance(my_meta.models["out.sin"], SurrogateModel)
    assert isinstance(my_meta.models["out.cos"], SurrogateModel)
    assert my_meta._trained

    class MetaB(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward("K")

    my_meta = MetaB("metaB", data=data)
    with pytest.raises(AttributeError):
        my_meta._train()

    data = pd.DataFrame(
        {"x": [(v, v ** 2) for v in x], "other": x, "sin": np.sin(x), "cos": np.cos(x)}
    )

    class MetaC(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_x")
            self.add_inward("var1", (0.0, 0.0))
            self.add_outward("K", 0.0)

    my_meta = MetaC(
        "metaA",
        data,
        mapping={"x": "inwards.var1", "other": "in_x.x", "sin": "outwards.K"},
    )
    # Should not raise
    my_meta._train()


def test_MetaSystem_add_output(test_data):
    class MetaB(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_output(OutPort, "out", model=FloatKrigingSurrogate)

    my_meta = MetaA("metaA", data=str(test_data / "export_doe.csv"))
    assert len(my_meta.models) == 2
    assert "out.sin" in my_meta.models
    assert "out.cos" in my_meta.models

    for model in my_meta.models.values():
        assert isinstance(model, NearestNeighbor)

    my_meta = MetaB("metaB", data=str(test_data / "export_doe.csv"))
    assert len(my_meta.models) == 2
    assert "out.sin" in my_meta.models
    assert "out.cos" in my_meta.models
    for model in my_meta.models.values():
        assert isinstance(model, FloatKrigingSurrogate)

    class MetaFail(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_output(OutPort, "out", model=str)

    with pytest.raises(TypeError):
        my_meta = MetaFail(data=str(test_data / "export_doe.csv"))


def test_MetaSystem_add_locals(test_data):
    class MetaFail(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward("K", model=str)

    with pytest.raises(TypeError):
        my_meta = MetaFail(data=str(test_data / "export_doe.csv"))

    class MetaC(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward("K3")

    my_meta = MetaC("metaC", data=str(test_data / "export_doe.csv"))
    assert "K3" in my_meta.outwards
    assert list(my_meta.models.keys()) == ["outwards.K3"]
    assert isinstance(my_meta.models["outwards.K3"], NearestNeighbor)

    class MetaD(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward("K3", model=FloatKrigingSurrogate)

    my_meta = MetaD("metaD", data=str(test_data / "export_doe.csv"))
    assert list(my_meta.models.keys()) == ["outwards.K3"]
    assert isinstance(my_meta.models["outwards.K3"], FloatKrigingSurrogate)

    class MetaE(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward({"K3": 1.0, "K": 0.0})

    my_meta = MetaE("metaE", data=str(test_data / "export_doe.csv"))
    assert len(my_meta.models) == 2
    assert "outwards.K3" in my_meta.models
    assert "outwards.K" in my_meta.models

    x = np.linspace(0, 10, 20)
    data = pd.DataFrame({"x": x, "K": [(np.sin(v), np.cos(v)) for v in x]})

    class MetaF(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward("K", [0.0, 0.0])

    my_meta = MetaF("metaF", data=data, mapping={"K": "outwards.K"})
    assert len(my_meta.models) == 2
    assert "outwards.K_index1" in my_meta.models
    assert "outwards.K_index0" in my_meta.models

    assert list(my_meta.outwards) == ["K", "K_index0", "K_index1"]

    class MetaG(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward("K3", [0.0, 0.0])

    with pytest.raises(AttributeError):
        MetaG("metaG", data=data)


def test_MetaSystem_rebuild_out_sequences():
    x = np.linspace(0, 10, 20)
    data = pd.DataFrame({"x": x, "K": [(np.sin(v), np.cos(v)) for v in x]})

    class MetaA(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward("K", [0.0, 0.0])

    my_meta = MetaA("metaA", data=data, mapping={"K": "outwards.K"})

    my_meta.K_index0 = 10.0
    my_meta.K_index1 = 11.0
    my_meta.rebuild_out_sequences()
    assert np.array_equal((10.0, 11.0), my_meta.K)

    data = pd.DataFrame({"x": x, "K": [(np.sin(v), np.cos(v), np.tan(v)) for v in x]})

    class MetaB(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_")
            self.add_outward("K", [0.0, 0.0, 0.0])

    my_meta = MetaB("metaB", data=data, mapping={"K": "outwards.K"})

    my_meta.K_index0 = 8.0
    my_meta.K_index1 = 15.0
    my_meta.K_index2 = 1.0
    my_meta.rebuild_out_sequences()
    assert np.array_equal([8.0, 15.0, 1.0], my_meta.K)


def test_MetaSystem_compute():
    x = np.linspace(0, 10, 20)
    data = pd.DataFrame({"x": x, "sin": np.sin(x), "cos": np.cos(x)})
    my_meta = MetaA(
        "metaA", data, mapping={"x": "in_.x", "sin": "out.sin", "cos": "out.cos"}
    )

    assert my_meta.out.sin == 1.0
    assert my_meta.out.cos == 1.0
    my_meta.in_.x = 2.1
    my_meta.compute()
    assert my_meta.out.sin == pytest.approx(0.8715856844, abs=1e-5)
    assert my_meta.out.cos == pytest.approx(-0.502574736, abs=1e-5)


def test_MetaSystem__get_models_inputs():
    class MetaA(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_x")
            self.add_input(YPort, "in_y")
            self.add_outward("K", 0.0)

    x = np.linspace(0, 10, 20)
    data = pd.DataFrame(
        {
            "x": x,
            "x_tuple": tuple((v, v) for v in x),
            "x_list": [(v, v, v) for v in x],
            "sin": np.sin(x),
            "cos": np.cos(x),
        }
    )

    my_meta = MetaA(
        "metaA", data, mapping={"x": "in_x.x", "sin": "out.sin", "cos": "out.cos"}
    )
    assert my_meta._input_names == ["in_x.x"]

    class MetaB(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_x")
            self.add_inward("var1", 0.0)
            self.add_inward("var2", (0.0, 0.0))
            self.add_outward("K", 0.0)

    my_meta = MetaB(
        "metaB",
        data,
        mapping={
            "x": "in_x.x",
            "x_tuple": "inwards.var2",
            "sin": "out.sin",
            "cos": "out.cos",
        },
    )
    assert set(my_meta._input_names) == {
        "in_x.x",
        "inwards.var1",
        "inwards.var2_index0",
        "inwards.var2_index1",
    }
    assert "var2_index0" in my_meta.inwards
    assert "var2_index1" in my_meta.inwards

    class MetaC(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_x")
            self.add_inward("var1", "a")
            self.add_inward("var2", (0.0, 0.0))
            self.add_outward("K", 0.0)

    my_meta = MetaC(
        "metaC",
        data,
        mapping={
            "x": "in_x.x",
            "x_tuple": "inwards.var2",
            "sin": "out.sin",
            "cos": "out.cos",
        },
    )
    assert set(my_meta._input_names) == {
        "in_x.x",
        "inwards.var2_index0",
        "inwards.var2_index1",
    }

    class MetaD(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_x")
            self.add_inward("var1", [0.0, 0.0, 0.0])
            self.add_inward("var2", (0.0, 0.0))
            self.add_outward("K", 0.0)

    my_meta = MetaD(
        "metaD",
        data,
        mapping={
            "x": "in_x.x",
            "x_tuple": "inwards.var2",
            "x_list": "inwards.var1",
            "sin": "out.sin",
            "cos": "out.cos",
        },
    )
    assert set(my_meta._input_names) == {
        "in_x.x",
        "inwards.var1_index0",
        "inwards.var1_index1",
        "inwards.var1_index2",
        "inwards.var2_index0",
        "inwards.var2_index1",
    }
    assert "var1_index0" in my_meta.inwards
    assert "inwards.var1_index0" in my_meta.training_data.columns
    assert "var1_index1" in my_meta.inwards
    assert "inwards.var1_index1" in my_meta.training_data.columns
    assert "var1_index2" in my_meta.inwards
    assert "inwards.var1_index2" in my_meta.training_data.columns


def test_MetaSystem_split_in_sequences():
    x = np.linspace(0, 10, 20)
    data = pd.DataFrame(
        {
            "x_tuple": [(v, v) for v in x],
            "x_list": [[v, v, v] for v in x],
            "sin": np.sin(x),
            "cos": np.cos(x),
        }
    )

    class MetaA(MetaSystem):
        def setup(self):
            self.add_input(XPort, "in_x")
            self.add_inward("var1", [0.0, 0.0, 0.0])
            self.add_inward("var2", (0.0, 0.0))
            self.add_outward("K", 0.0)

    my_meta = MetaA(
        "metaA",
        data,
        mapping={"x_list": "inwards.var1", "x_tuple": "inwards.var2", "cos": "out.cos"},
    )

    my_meta.var1 = [10.0, 15.1, 50.2]
    my_meta.split_in_sequences()
    assert my_meta.var1 == [
        my_meta.var1_index0,
        my_meta.var1_index1,
        my_meta.var1_index2,
    ]

    my_meta.var2 = (1.1, 9.0)
    my_meta.split_in_sequences()
    assert my_meta.var2 == (my_meta.var2_index0, my_meta.var2_index1)


def test_MetaSystem_overall():
    x = np.linspace(0, 10, 20)
    data = pd.DataFrame(
        {"in": [(v, v ** 2) for v in x], "out": [(3.0 * v, 5.0 * v ** 2) for v in x]}
    )

    class MetaA(MetaSystem):
        def setup(self):
            self.add_inward("var_in", [0.0, 0])
            self.add_outward("var_out", [0.0, 0.0], model=FloatKrigingSurrogate)

    my_meta = MetaA(
        "metaA", data, mapping={"in": "inwards.var_in", "out": "outwards.var_out"}
    )
    assert {"outwards.var_out_index0", "outwards.var_out_index1"} == set(
        my_meta.models.keys()
    )
    my_meta.compute()

    my_meta.var_in = [10.0, 100.0]
    my_meta.compute()

    assert my_meta.var_in_index0 == pytest.approx(10.0)
    assert my_meta.var_in_index1 == pytest.approx(100.0)

    assert my_meta.var_out[0] == pytest.approx(30.0)
    assert my_meta.var_out[1] == pytest.approx(500.0)

    np.array_equal((my_meta.var_out_index0, my_meta.var_out_index1), my_meta.var_out)
