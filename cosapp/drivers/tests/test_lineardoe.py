import numpy as np
import pytest

from cosapp.drivers import LinearDoE, RunOnce
from cosapp.recorders import DataFrameRecorder
from cosapp.tests.library.systems import Multiply2


def test_LinearDoE_init():
    s = Multiply2("mult")
    d = s.add_driver(LinearDoE("doe"))
    assert isinstance(d, LinearDoE)


def test_LinearDoE_add_input_var():
    s = Multiply2("mult")
    d = s.add_driver(LinearDoE("doe"))

    d.add_input_var("K1", 0.0, 10.0)
    assert d.input_vars == {"K1": {"lower": 0.0, "upper": 10.0, "count": 2}}

    d.add_input_var({"K1": {"lower": 0.2, "upper": 20.0}})
    assert d.input_vars == {"K1": {"lower": 0.2, "upper": 20.0, "count": 2}}

    d.add_input_var(
        {"K1": {"lower": 0.2, "upper": 20.0}, "K2": {"lower": 0.1, "upper": 200.0}}
    )
    assert d.input_vars == {
        "K1": {"lower": 0.2, "upper": 20.0, "count": 2},
        "K2": {"lower": 0.1, "upper": 200.0, "count": 2},
    }

    with pytest.raises(TypeError):
        d.add_input_var({"myvar": {"upper": 20.0}})
    with pytest.raises(TypeError):
        d.add_input_var({"myvar": 20.0})

    with pytest.raises(AttributeError):
        d.add_input_var("myvar", 0.0, 10.0)
    with pytest.raises(AttributeError):
        d.add_input_var({"myvar": {"lower": 0.2, "upper": 20.0}})


def test_LinearDoE__build_cases():
    s = Multiply2("mult")
    d = s.add_driver(LinearDoE("doe"))

    d._build_cases()
    assert d.cases == [()]

    d.add_input_var("K1", lower=0.2, upper=20.0)
    d.add_input_var("K2", lower=0.1, upper=200.0)
    d._build_cases()
    assert np.allclose(
        np.asarray(d.cases),
        np.asarray([(0.2, 0.1), (0.2, 200.0), (20.0, 0.1), (20.0, 200.0)]),
        rtol=1e-9,
    )

    d.add_input_var("K1", lower=0.2, upper=20.0, count=3)
    d.add_input_var("K2", lower=0.1, upper=200.0)
    d._build_cases()
    assert np.allclose(
        np.asarray(d.cases),
        np.asarray(
            [
                (0.2, 0.1),
                (0.2, 200.0),
                (10.1, 0.1),
                (10.1, 200.0),
                (20.0, 0.1),
                (20.0, 200.0),
            ]
        ),
        rtol=1e-9,
    )


def test_LinearDoE_compute():
    s = Multiply2("mult")
    d = s.add_driver(LinearDoE("doe"))
    d.add_child(RunOnce("run"))
    d.add_recorder(DataFrameRecorder(includes=["K?", "p_out.x"], raw_output=True))

    d.add_input_var(
        {
            "K1": {"lower": 0.0, "upper": 20.0, "count": 3},
            "K2": {"lower": 0.0, "upper": 200.0, "count": 3},
        }
    )
    d.run_once()

    assert s.K1 == 20.0
    assert s.K2 == 200.0
    assert len(d.recorder.data) == 9
    assert d.recorder.data.iloc[4]["K1"] == 10.0
    assert d.recorder.data.iloc[4]["K2"] == 100.0
    assert d.recorder.data.iloc[4]["p_out.x"] == 1000.0
    assert d.recorder.data.iloc[8]["K1"] == 20.0
    assert d.recorder.data.iloc[8]["K2"] == 200.0
    assert d.recorder.data.iloc[8]["p_out.x"] == 4000.0


def test_LinearDoE_run_once():
    s = Multiply2("mult")
    d = s.add_driver(LinearDoE("doe"))
    d.add_child(RunOnce("run"))
    d.add_recorder(DataFrameRecorder(includes=["p_*.x", "K?"], raw_output=True))

    d.add_input_var(
        {
            "K1": {"lower": 0.0, "upper": 20.0, "count": 3},
            "K2": {"lower": 0.0, "upper": 200.0, "count": 3},
        }
    )
    d.run_once()
    assert d.recorder.data.shape == (9, 4 + len(DataFrameRecorder.SPECIALS))
