import pytest

import pandas as pd
from cosapp.drivers import NonLinearSolver, RunSingleCase, NonLinearMethods, RunOnce
from cosapp.recorders.recorder import BaseRecorder
from cosapp.recorders.dataframe_recorder import DataFrameRecorder
from cosapp.tests.library.systems.vectors import AllTypesSystem, BooleanSystem
from cosapp.tests.library.systems.multiply import Multiply2


def test_DataframeRecorder___init__default():
    """Test default constructor"""
    recorder = DataFrameRecorder()
    # Members inherited from base class
    assert recorder.includes == ["*"]
    assert recorder.excludes == []
    assert recorder.hold == False
    assert recorder._raw_output == True
    assert recorder._numerical_only == False
    assert recorder.precision == 9
    assert recorder.section == ""
    assert recorder.watched_object is None
    # Specific property 'data'
    assert isinstance(recorder.data, pd.DataFrame)
    assert len(recorder.data) == 0

def test_DataframeRecorder_start():
    recorder = DataFrameRecorder(raw_output=True)
    s = AllTypesSystem("sub")
    t = AllTypesSystem("top")
    t.add_child(s)
    recorder.watched_object = t

    recorder.start()
    recorder.record_state(0)
    assert recorder.data.shape == (1, len(BaseRecorder.SPECIALS) + len(recorder.get_variables_list()))
    assert set(recorder.data.columns) == set([*BaseRecorder.SPECIALS, *recorder.get_variables_list()])

    recorder.record_state(1)
    assert recorder.data.shape == (2, len(BaseRecorder.SPECIALS) + len(recorder.get_variables_list()))
    assert set(recorder.data.columns) == set([*BaseRecorder.SPECIALS, *recorder.get_variables_list()])

    recorder.start()
    recorder.record_state(0)
    assert recorder.data.shape == (1, len(BaseRecorder.SPECIALS) + len(recorder.get_variables_list()))
    assert set(recorder.data.columns) == set([*BaseRecorder.SPECIALS, *recorder.get_variables_list()])

    # Test hold
    recorder.hold = True
    recorder.start()
    recorder.record_state(1)
    assert recorder.data.shape == (2, len(BaseRecorder.SPECIALS) + len(recorder.get_variables_list()))
    assert set(recorder.data.columns) == set([*BaseRecorder.SPECIALS, *recorder.get_variables_list()])


def test_DataframeRecorder_record_iteration():
    recorder = DataFrameRecorder(raw_output=False)
    s = AllTypesSystem("sub")
    t = AllTypesSystem("top")
    t.add_child(s)
    recorder.watched_object = t

    recorder.record_state(0)
    assert recorder.data.shape == (1, len(BaseRecorder.SPECIALS) + len(recorder.get_variables_list()))
    headers = ["Section", "Reference", "Status", "Error code"]
    headers.extend(
        [
            "{} [{}]".format(n, u)
            for n, u in zip(recorder.get_variables_list(), recorder._get_units(recorder.get_variables_list()))
        ]
    )
    assert set(recorder.data.columns) == set(headers)

    recorder = DataFrameRecorder(raw_output=True)
    s = AllTypesSystem("sub")
    t = AllTypesSystem("top")
    t.add_child(s)
    recorder.watched_object = t

    recorder.record_state(0)
    assert recorder.data.shape == (1, len(BaseRecorder.SPECIALS) + len(recorder.get_variables_list()))
    assert set(recorder.data.columns) == set([*BaseRecorder.SPECIALS, *recorder.get_variables_list()])

    recorder.record_state(1)
    assert recorder.data.shape == (2, len(BaseRecorder.SPECIALS) + len(recorder.get_variables_list()))
    assert set(recorder.data.columns) == set([*BaseRecorder.SPECIALS, *recorder.get_variables_list()])


def test_DataframeRecorder_restore():
    s = Multiply2("mult2")
    solve = s.add_driver(NonLinearSolver("solve", method=NonLinearMethods.NR))
    design = solve.add_child(RunSingleCase("design"))
    recorder = solve.add_recorder(DataFrameRecorder(hold=True))

    design.design.add_unknown("p_in.x").add_equation("p_out.x == 10")
    s.run_drivers()

    s.K1 = 1
    s.run_drivers()

    s.K1 = 2
    s.run_drivers()

    with pytest.raises(TypeError):
        recorder.restore(1.)
    with pytest.raises(TypeError):
        recorder.restore("fake_index")

    assert s.K1 == 2
    assert s.p_in.x == pytest.approx(1, rel=1e-4)

    recorder.restore(0)
    assert s.K1 == 5
    assert s.p_in.x == pytest.approx(0.4, rel=1e-4)

    recorder.restore(1)
    assert s.K1 == 1
    assert s.p_in.x == pytest.approx(2, rel=1e-4)

    s.run_drivers()
    assert s.K1 == 1
    assert s.p_in.x == pytest.approx(2, rel=1e-4)

    s = BooleanSystem("s")
    run = s.add_driver(RunOnce("run", method=NonLinearMethods.NR))
    recorder = run.add_recorder(DataFrameRecorder(hold=True))

    s.run_drivers()
    s.a = False
    recorder.restore(0)
    assert s.a == True
