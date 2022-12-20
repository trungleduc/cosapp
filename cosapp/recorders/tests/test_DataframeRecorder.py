import pytest

import numpy as np
import pandas as pd
from cosapp.ports import Port
from cosapp.systems import System
from cosapp.drivers import NonLinearSolver, RunSingleCase, NonLinearMethods, RunOnce, EulerExplicit
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
    # check exported data
    data = recorder.export_data()
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 0


def test_DataframeRecorder_start():
    recorder = DataFrameRecorder(raw_output=True)
    s = AllTypesSystem("sub")
    t = AllTypesSystem("top")
    t.add_child(s)
    recorder.watched_object = t

    recorder.start()
    recorder.record_state(0)
    data = recorder.export_data()
    assert data.shape == (1, len(BaseRecorder.SPECIALS) + len(recorder.field_names()))
    assert set(data.columns) == set([*BaseRecorder.SPECIALS, *recorder.field_names()])

    recorder.record_state(1)
    data = recorder.export_data()
    assert data.shape == (2, len(BaseRecorder.SPECIALS) + len(recorder.field_names()))
    assert set(data.columns) == set([*BaseRecorder.SPECIALS, *recorder.field_names()])

    recorder.start()
    recorder.record_state(0)
    data = recorder.export_data()
    assert data.shape == (1, len(BaseRecorder.SPECIALS) + len(recorder.field_names()))
    assert set(data.columns) == set([*BaseRecorder.SPECIALS, *recorder.field_names()])

    # Test hold
    recorder.hold = True
    recorder.start()
    recorder.record_state(1)
    data = recorder.export_data()
    assert data.shape == (2, len(BaseRecorder.SPECIALS) + len(recorder.field_names()))
    assert set(data.columns) == set([*BaseRecorder.SPECIALS, *recorder.field_names()])


def test_DataframeRecorder_record_iteration():
    recorder = DataFrameRecorder(raw_output=False)
    s = AllTypesSystem("sub")
    t = AllTypesSystem("top")
    t.add_child(s)
    recorder.watched_object = t

    recorder.record_state(0)
    data = recorder.export_data()
    assert data.shape == (1, len(BaseRecorder.SPECIALS) + len(recorder.field_names()))
    headers = ["Section", "Reference", "Status", "Error code"]
    headers.extend(
        [
            f"{n} [{u}]"
            for n, u in zip(recorder.field_names(), recorder._get_units(recorder.field_names()))
        ]
    )
    assert set(data.columns) == set(headers)

    recorder = DataFrameRecorder(raw_output=True)
    s = AllTypesSystem("sub")
    t = AllTypesSystem("top")
    t.add_child(s)
    recorder.watched_object = t

    recorder.record_state(0)
    data = recorder.export_data()
    assert data.shape == (1, len(BaseRecorder.SPECIALS) + len(recorder.field_names()))
    assert set(data.columns) == set([*BaseRecorder.SPECIALS, *recorder.field_names()])

    recorder.record_state(1)
    data = recorder.export_data()
    assert data.shape == (2, len(BaseRecorder.SPECIALS) + len(recorder.field_names()))
    assert set(data.columns) == set([*BaseRecorder.SPECIALS, *recorder.field_names()])


def test_DataframeRecorder_record_properties(SystemWithProps):
    """Test a recorder recording system and port properties"""
    s = SystemWithProps('s')
    driver = s.add_driver(EulerExplicit(dt=0.1, time_interval=(0, 1)))

    driver.set_scenario(
        values={
            'in_.x': 'cos(pi * t)',
            'in_.y': '1 + 0.5 * sin(2 * pi * t)',
        },
    )
    driver.add_recorder(
        DataFrameRecorder(includes=['a', '*_ratio'])
    )

    s.run_drivers()
    df = driver.recorder.export_data()
    headers = ["Section", "Reference", "Status", "Error code"]
    headers.extend(['time', 'a', 'in_.xy_ratio', 'out.xy_ratio', 'bogus_ratio'])
    assert set(df.columns) == set(headers)
    
    time = np.asarray(df['time'])
    out_xy_ratio = np.asarray(df['out.xy_ratio'])
    assert time == pytest.approx(np.linspace(0, 1, 11))
    assert out_xy_ratio == pytest.approx(np.cos(np.pi * time) / (2 + np.sin(2 * np.pi * time)))
    assert np.asarray(df['a']) == pytest.approx(0.1 * out_xy_ratio)
    assert np.asarray(df['bogus_ratio']) == pytest.approx(2 * np.cos(np.pi * time))


def test_DataframeRecorder_record_expressions(SystemWithProps):
    """Test a recorder involving evaluable expressions"""
    s = SystemWithProps('s')
    driver = s.add_driver(EulerExplicit(dt=0.1, time_interval=(0, 1)))

    driver.set_scenario(
        values={
            'in_.x': 'cos(pi * t)',
            'in_.y': '1 + 0.5 * sin(2 * pi * t)',
        },
    )
    driver.add_recorder(
        DataFrameRecorder(
            includes=['a', 'bogus*', '-2 * a + out.y', 'sin(pi * t)'],
        )
    )

    s.run_drivers()
    df = driver.recorder.export_data()
    headers = ["Section", "Reference", "Status", "Error code"]
    headers.extend(['time', 'a', '-2 * a + out.y', 'sin(pi * t)', 'bogus_ratio'])
    assert set(df.columns) == set(headers)
    
    time = np.asarray(df['time'])
    assert time == pytest.approx(np.linspace(0, 1, 11))
    exact = {
        'a': lambda t: 0.1 * np.cos(np.pi * t) / (2 + np.sin(2 * np.pi * t)),
        'out.y': lambda t: 2 + np.sin(2 * np.pi * t),
    }
    expected = {
        'a': pytest.approx(exact['a'](time), rel=1e-14),
        'sin(pi * t)': pytest.approx(np.sin(np.pi * time), rel=1e-14),
        'bogus_ratio': pytest.approx(2 * np.cos(np.pi * time), rel=1e-14),
        '-2 * a + out.y': pytest.approx(-2 * exact['a'](time) + exact['out.y'](time), rel=1e-14),
    }
    for field in expected:
        assert np.asarray(df[field]) == expected[field], f"field: {field}"


def test_DataframeRecorder_restore():
    s = Multiply2("mult2")
    solver = s.add_driver(NonLinearSolver("solver", method=NonLinearMethods.NR))
    solver.add_unknown("p_in.x").add_equation("p_out.x == 10")

    recorder = solver.add_recorder(DataFrameRecorder(hold=True))
    
    s.run_drivers()

    s.K1 = 1
    s.run_drivers()

    s.K1 = 2
    s.run_drivers()

    with pytest.raises(TypeError):
        recorder.restore(1.)

    with pytest.raises(TypeError):
        recorder.restore("fake_index")

    with pytest.raises(IndexError):
        recorder.restore(-1)

    with pytest.raises(IndexError):
        recorder.restore(3)

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


def test_DataframeRecorder_check():
    import copy

    class NonCopyable:
        def __deepcopy__(self, memo=None):
            raise copy.Error(self.__class__.__name__)
    
    class LocalSystem(System):
        def setup(self):
            self.add_inward('foo', NonCopyable())

    with pytest.raises(copy.Error):  # sanity check
        foo = NonCopyable()
        copy.deepcopy(foo)

    recorder = DataFrameRecorder()
    recorder.watched_object = LocalSystem('s')

    with pytest.warns(RuntimeWarning, match=r"Captured exception .* while trying to collect data"):
        recorder.check()

    with pytest.raises(TypeError, match="Cannot record s.foo"):
        recorder.record_state(0)
