import pytest
from unittest import mock

from cosapp.core.signal import Slot
from cosapp.recorders.recorder import BaseRecorder


@pytest.fixture(autouse=True)
def PatchBaseRecorder():
    """Patch BaseRecorder to make it instanciable for tests"""
    patcher = mock.patch.multiple(
        BaseRecorder,
        __abstractmethods__ = set(),
        formatted_data = lambda self: list(),
    )
    patcher.start()
    yield
    patcher.stop()


@pytest.mark.parametrize("kwargs, expected", [
    (dict(), dict(includes=["*"], excludes=[])),  # default constructor
    (dict(includes=[], excludes=[]), dict(includes=[], excludes=[])),
    (dict(includes=('q', 'c')), dict(includes=['q', 'c'], excludes=[])),
    (dict(includes='*c', excludes='f?'), dict(includes=['*c'], excludes=['f?'])),
    (dict(
        includes=['a', 'b?', '*c'],
        excludes=['d', '*e', 'f?'],
        section='banana',
        precision=12,
        hold=True,
        raw_output=True,
        ), 
        dict()),
    (
        dict(excludes=dict(cool=True), includes=set('abracadabra')),
        dict(includes=['a', 'b', 'c', 'd', 'r'], excludes=['cool'])
    ),
    # Erroneous cases:
    (dict(includes=('q', 2)), dict(error=TypeError, match="'includes' must be a string, or a sequence of strings")),
    (dict(includes=23), dict(error=TypeError, match="'includes' must be a string, or a sequence of strings")),
    (dict(excludes=23), dict(error=TypeError, match="'excludes' must be a string, or a sequence of strings")),
    (dict(section=23), dict(error=TypeError, match="'section' should be str")),
    (dict(hold='yes'), dict(error=TypeError, match="'hold' should be bool")),
    (dict(precision='yes'), dict(error=TypeError, match="'precision' should be int")),
    (dict(precision=-2), dict(error=ValueError)),
    (dict(precision=0), dict(error=ValueError)),
    (dict(raw_output=0), dict(error=TypeError, match="'raw_output' should be bool")),
    (dict(numerical_only=0), dict(error=TypeError, match="'numerical_only' should be bool")),
])
def test_BaseRecorder__init__(kwargs, expected):
    error = expected.get('error', None)
    if error is None:
        if len(expected) == 0:
            expected = kwargs.copy()
        recorder = BaseRecorder(**kwargs)
        assert set(recorder.includes) == set(expected['includes'])
        assert set(recorder.excludes) == set(expected['excludes'])
        assert recorder.hold == expected.get('hold', False)
        assert recorder._raw_output == expected.get('raw_output', False)
        assert recorder._numerical_only == expected.get('numerical_only', False)
        assert recorder.precision == expected.get('precision', 9)
        assert recorder.section == expected.get('section', '')
        assert recorder.watched_object is None
    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            BaseRecorder(**kwargs)


@pytest.mark.parametrize("value, expected", [
    ("section 1", dict()),
    ("foo.bar", dict()),
    (42, dict(error=TypeError, match="'section' should be str")),
    (True, dict(error=TypeError, match="'section' should be str")),
])
def test_BaseRecorder_section(value, expected):
    recorder = BaseRecorder()
    assert recorder.section == ""

    error = expected.get('error', None)
    if error is None:
        recorder.section = value
        assert recorder.section == expected.get('value', value)
    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            recorder.section = value


@pytest.mark.parametrize("value, expected", [
    (True, dict()),
    (False, dict()),
    (42, dict(error=TypeError, match="'hold' should be bool")),
    ("yes", dict(error=TypeError, match="'hold' should be bool")),
    ("True", dict(error=TypeError, match="'hold' should be bool")),
])
def test_BaseRecorder_hold(value, expected):
    recorder = BaseRecorder()
    assert recorder.hold is False

    error = expected.get('error', None)
    if error is None:
        recorder.hold = value
        assert recorder.hold == expected.get('value', value)
    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            recorder.hold = value


@pytest.mark.parametrize("value, expected", [
    (4, dict()),
    (42, dict()),
    (0, dict(error=ValueError, match="'precision'.*invalid")),
    (-2, dict(error=ValueError, match="'precision'.*invalid")),
    (1.2, dict(error=TypeError, match="'precision' should be int")),
    ("5", dict(error=TypeError, match="'precision' should be int")),
])
def test_BaseRecorder_precision(value, expected):
    recorder = BaseRecorder()
    assert recorder.precision == 9

    error = expected.get('error', None)
    if error is None:
        recorder.precision = value
        assert recorder.precision == expected.get('value', value)
    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            recorder.precision = value


def test_BaseRecorder_watched_object(AllTypesSystem, SystemWithProps):
    recorder = BaseRecorder(includes='?')
    assert recorder.watched_object is None

    a = AllTypesSystem('a')
    recorder.watched_object = a
    assert recorder.watched_object is a
    assert set(recorder.field_names()) == set('abcde')

    with pytest.raises(TypeError, match="Recorder must be attached to a System or a Driver"):
        recorder.watched_object = 'dummy'

    # Watch new object; should update field names
    p = SystemWithProps('p')
    recorder.watched_object = p
    assert recorder.watched_object is p
    assert set(recorder.field_names()) == {'a'}


def test_BaseRecorder_includes():
    recorder = BaseRecorder()
    assert recorder.includes == ["*"]
    # Other values tested in test_BaseRecorder___init__
    with pytest.raises(AttributeError):
        recorder.includes = 'dummy'


def test_BaseRecorder_excludes():
    recorder = BaseRecorder()
    assert recorder.excludes == list()
    # Other values tested in test_BaseRecorder___init__
    with pytest.raises(AttributeError):
        recorder.excludes = 'dummy'


def test_BaseRecorder_field_names(AllTypesSystem):
    recorder = BaseRecorder()
    assert recorder.field_names() == list()

    # Single system
    s = AllTypesSystem('test')
    recorder.watched_object = s
    assert set(recorder.field_names()) == {'a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x'}

    # Two levels system
    recorder = BaseRecorder()
    t = AllTypesSystem('top')
    t.add_child(s)
    recorder.watched_object = t
    assert set(recorder.field_names()) == {
        'a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x',
        'test.a', 'test.b', 'test.c', 'test.d', 'test.e',
        'test.in_.x', 'test.out.x',
    }


@pytest.mark.parametrize("includes, expected", [
    ('sub.a', ['sub.a']),
    ('sub.d', ['sub.d']),
    ('sub.inwards.a', ['sub.a']),
    ('inwards.a', ['a']),
    ('sub.outwards.d', ['sub.d']),
    ('outwards.d', ['d']),
    ('sub.?', ['sub.a', 'sub.b', 'sub.c', 'sub.e', 'sub.d']),
    ('sub.*', ['sub.in_.x', 'sub.a', 'sub.b', 'sub.c', 'sub.e', 'sub.d', 'sub.out.x']),
    (['sub.*', '*d', 'a'], ['a', 'd', 'sub.in_.x', 'sub.a', 'sub.b', 'sub.c', 'sub.e', 'sub.d', 'sub.out.x']),
    ('banana', []),
])
def test_BaseRecorder_field_names__includes(AllTypesSystem, includes, expected):
    s = AllTypesSystem('sub')
    t = AllTypesSystem('top')
    t.add_child(s)
    recorder = BaseRecorder(includes=includes)
    recorder.watched_object = t
    assert recorder.watched_object is t
    assert set(recorder.field_names()) == set(expected)  # test lists regardless of order


@pytest.mark.parametrize("excludes, expected", [
    ('*', []),
    ('sub.a', ['in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
                'sub.b', 'sub.c', 'sub.d', 'sub.e',
                'sub.in_.x', 'sub.out.x']),
    ('sub.inwards.a', ['in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
                        'sub.b', 'sub.c', 'sub.d', 'sub.e',
                        'sub.in_.x', 'sub.out.x']),
    ('inwards.a', ['in_.x', 'out.x', 'sub.a', 'b', 'c', 'e', 'd',
                   'sub.b', 'sub.c', 'sub.d', 'sub.e',
                   'sub.in_.x', 'sub.out.x']),
    ('sub.d', ['in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
                'sub.a', 'sub.b', 'sub.c', 'sub.e',
                'sub.in_.x', 'sub.out.x']),
    ('sub.outwards.d', ['in_.x', 'out.x', 'a', 'b', 'c', 'e', 'd',
                         'sub.a', 'sub.b', 'sub.c', 'sub.e',
                         'sub.in_.x', 'sub.out.x']),
    ('outwards.d', ['in_.x', 'out.x', 'a', 'b', 'c', 'e',
                    'sub.d', 'sub.a', 'sub.b', 'sub.c', 'sub.e',
                    'sub.in_.x', 'sub.out.x']),
    ('sub.?', ['a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x', 'sub.in_.x', 'sub.out.x']),
    ('sub.*', ['a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x']),
    (['sub.*', '*d', 'a'], ['b', 'c', 'e', 'in_.x', 'out.x']),
    ('banana', ['a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x',
                'sub.in_.x', 'sub.a', 'sub.b', 'sub.c', 'sub.e',
                'sub.d', 'sub.out.x']),
])
def test_BaseRecorder_field_names__excludes(AllTypesSystem, excludes, expected):
    s = AllTypesSystem('sub')
    t = AllTypesSystem('top')
    t.add_child(s)
    recorder = BaseRecorder(excludes=excludes)
    recorder.watched_object = t
    assert recorder.watched_object is t
    assert set(recorder.field_names()) == set(expected)  # test lists regardless of order


@pytest.mark.parametrize("ctor, expected", [
    (dict(includes='?'), list('abcde')),
    (dict(includes='?', excludes='a'), list('bcde')),
    (dict(includes=['a', 'b'], excludes='a'), ['b']),
    (
        dict(includes=['a', '*_ratio']),
        ['a', 'sub.in_.xy_ratio', 'sub.out.xy_ratio', 'sub.bogus_ratio'],
    ),
    (
        dict(includes=['a', '*_ratio'], excludes=['*bogus*']),
        ['a', 'sub.in_.xy_ratio', 'sub.out.xy_ratio'],
    ),
    (dict(includes=['sub.a']), ['sub.a']),
    (dict(includes=['sub.a'], excludes='sub.?'), []),
    (dict(includes=['a[-1]'], excludes='?'), ['a[-1]']),
    (dict(includes=['a[-1]', '2 * a + sub.out.y']), ['a[-1]', '2 * a + sub.out.y']),
    (dict(includes=['sub.a +']), []),    # expressions with syntax errors should be filtered out
    (dict(includes=['2 * sub.?']), []),  # can't combine mathematical expression and search pattern
])
def test_BaseRecorder_field_names_expressions(AllTypesSystem, SystemWithProps, ctor, expected):
    """Test inclusion and exclusion patterns involving properties and evaluable expressions"""
    top = AllTypesSystem('top')
    sub = SystemWithProps('sub')
    top.add_child(sub)
    recorder = BaseRecorder(**ctor)
    recorder.watched_object = top
    assert recorder.watched_object is top

    assert set(recorder.field_names()) == set(expected)
    # Test `in` operator:
    assert all(field in recorder for field in expected)


@pytest.mark.parametrize("info, expected", [
    (dict(includes=None, excludes=None), list('bcde')),
    (dict(includes='?'), list('bcde')),
    (dict(includes='a'), list('bcde')),
    (dict(includes=['x', 'y'], excludes='b'), list('cde')),
    (
        dict(includes='sub.*', excludes='*_ratio'),
        list('bcde') + ['sub.a', 'sub.in_.x', 'sub.in_.y', 'sub.out.x', 'sub.out.y'],
    ),
    (
        dict(includes='*_ratio'),
        list('bcde') + ['sub.in_.xy_ratio', 'sub.out.xy_ratio', 'sub.bogus_ratio'],
    ),
    (
        dict(includes=['*_ratio'], excludes=['*bogus*']),
        list('bcde') + ['sub.in_.xy_ratio', 'sub.out.xy_ratio'],
    ),
    (dict(includes=['sub.a']), list('bcde') + ['sub.a']),
    (dict(includes=['sub.?'], excludes='sub.a'), list('bcde')),
    (dict(includes=['sub.?'], excludes='sub.b'), list('bcde') + ['sub.a']),
    (dict(includes=['a[-1]']), list('bcde') + ['a[-1]']),
    (dict(includes=['a[-1]'], excludes='?'), ['a[-1]']),
    (dict(includes=['a[-1]', '2 * a + sub.out.y'], excludes='?'), ['a[-1]', '2 * a + sub.out.y']),
])
def test_BaseRecorder_extend(AllTypesSystem, SystemWithProps, info, expected):
    """Test factory `extend`"""
    top = AllTypesSystem('top')
    sub = SystemWithProps('sub')
    top.add_child(sub)

    recorder = BaseRecorder(includes='?', excludes='a')
    recorder.watched_object = top
    assert recorder.watched_object is top
    assert set(recorder.field_names()) == set('bcde')

    # Create extended recorder
    newrec = BaseRecorder.extend(recorder, **info)
    assert isinstance(newrec, BaseRecorder)
    assert newrec.watched_object is top
    assert set(newrec.field_names()) == set(expected)
    for attr in (
        'section',
        'precision',
        'hold',
        '_raw_output',
        '_numerical_only',
    ):
        assert getattr(newrec, attr) == getattr(recorder, attr), f"attribute {attr!r}"


def test_BaseRecorder__get_units(AllTypesSystem):
    recorder = BaseRecorder()
    assert recorder._get_units() == list()

    # Single system
    s = AllTypesSystem('test')
    recorder.watched_object = s
    units = recorder._get_units()
    assert len(units) == len(recorder.field_names())
    assert all(isinstance(unit, str) for unit in units)

    # Two levels system
    recorder = BaseRecorder()
    t = AllTypesSystem('top')
    t.add_child(s)
    recorder.watched_object = t
    units = recorder._get_units()
    assert len(units) == len(recorder.field_names())
    assert all(isinstance(unit, str) for unit in units)

    # Test includes
    recorder = BaseRecorder(includes='test.*')
    recorder.watched_object = t
    units = recorder._get_units()
    assert len(units) == len(recorder.field_names())
    assert all(isinstance(unit, str) for unit in units)

    recorder = BaseRecorder(includes='banana')
    recorder.watched_object = t
    units = recorder._get_units()
    assert len(units) == 0
    assert len(units) == len(recorder.field_names())

    # Test excludes
    recorder = BaseRecorder(excludes='test.*')
    recorder.watched_object = t
    units = recorder._get_units()
    assert len(units) == len(recorder.field_names())
    assert all(isinstance(unit, str) for unit in units)

    recorder = BaseRecorder(excludes='*')
    recorder.watched_object = t
    units = recorder._get_units()
    assert len(units) == len(recorder.field_names())
    assert all(isinstance(unit, str) for unit in units)

    recorder = BaseRecorder(excludes='banana')
    recorder.watched_object = t
    units = recorder._get_units()
    assert len(units) == len(recorder.field_names())
    assert all(isinstance(unit, str) for unit in units)


def test_BaseRecorder_start():
    recorder = BaseRecorder()
    with pytest.raises(RuntimeError, match='A recorder should be watching a Driver'):
        recorder.start()


def test_BaseRecorder_record_state():
    with mock.patch("cosapp.core.signal.signal.inspect"):
        fake_callback = mock.Mock(spec=lambda **kwargs: None)
        fake_callback.return_value = None

        recorder = BaseRecorder()
        recorder.state_recorded.connect(Slot(fake_callback))
        recorder.record_state(time_ref='time', status='OK', error_code='000')

        fake_callback.assert_called_once_with(time_ref='time', status='OK', error_code='000')


def test_BaseRecorder_clear():
    with mock.patch("cosapp.core.signal.signal.inspect"):
        fake_callback = mock.Mock(spec=lambda **kwargs: None)
        fake_callback.return_value = None

        recorder = BaseRecorder()
        recorder.cleared.connect(Slot(fake_callback))
        recorder.record_state(time_ref='time', status='OK', error_code='000')
        recorder.clear()
        fake_callback.assert_called_once_with()


def test_BaseRecorder_data_warning():
    recorder = BaseRecorder()
    with pytest.warns(DeprecationWarning, match="use export_data()"):
        recorder.data
