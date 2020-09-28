import pytest
from unittest import mock

from cosapp.core.signal import Slot
from cosapp.recorders.recorder import BaseRecorder


class MockupRecorder(BaseRecorder):
    """Mock-up class to test BaseRecorder"""

    @property
    def _raw_data(self):
        return list()

    def exit(self):
        pass


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
    # Erroneous cases:
    (dict(includes=('q', 2)), dict(error=TypeError, match="'includes' must be a string, or a sequence of strings")),
    (dict(excludes={'q': 'e'}), dict(error=TypeError, match="'excludes' must be a string, or a sequence of strings")),
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
        recorder = MockupRecorder(**kwargs)
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
            MockupRecorder(**kwargs)


@pytest.mark.parametrize("value, expected", [
    ("section 1", dict()),
    ("foo.bar", dict()),
    (42, dict(error=TypeError, match="'section' should be str")),
    (True, dict(error=TypeError, match="'section' should be str")),
])
def test_BaseRecorder_section(value, expected):
    recorder = MockupRecorder()
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
    recorder = MockupRecorder()
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
    recorder = MockupRecorder()
    assert recorder.precision == 9

    error = expected.get('error', None)
    if error is None:
        recorder.precision = value
        assert recorder.precision == expected.get('value', value)
    else:
        pattern = expected.get('match', None)
        with pytest.raises(error, match=pattern):
            recorder.precision = value


def test_BaseRecorder_watched_object(AllTypesSystem):
    recorder = MockupRecorder()
    assert recorder.watched_object is None

    s = AllTypesSystem('test')
    recorder.watched_object = s
    assert recorder.watched_object is s

    with pytest.raises(TypeError, match="Record must be attached to a Driver"):
        recorder.watched_object = 'dummy'


def test_BaseRecorder_includes():
    recorder = MockupRecorder()
    assert recorder.includes == ["*"]
    # Other values tested in test_BaseRecorder___init__
    with pytest.raises(AttributeError):
        recorder.includes = 'dummy'


def test_BaseRecorder_excludes():
    recorder = MockupRecorder()
    assert recorder.excludes == list()
    # Other values tested in test_BaseRecorder___init__
    with pytest.raises(AttributeError):
        recorder.excludes = 'dummy'


def test_BaseRecorder_get_variables_list(AllTypesSystem):
    recorder = MockupRecorder()
    assert recorder.get_variables_list() == list()

    # Single system
    s = AllTypesSystem('test')
    recorder.watched_object = s
    assert set(recorder.get_variables_list()) == set(('a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x'))

    # Two levels system
    recorder = MockupRecorder()
    t = AllTypesSystem('top')
    t.add_child(s)
    recorder.watched_object = t
    assert set(recorder.get_variables_list()) == set(('a', 'b', 'c', 'e', 'd', 'in_.x', 'out.x',
                                                'test.a', 'test.b', 'test.c', 'test.d', 'test.e',
                                                'test.in_.x', 'test.out.x'))


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
def test_BaseRecorder_get_variables_list__includes(AllTypesSystem, includes, expected):
    s = AllTypesSystem('sub')
    t = AllTypesSystem('top')
    t.add_child(s)
    recorder = MockupRecorder(includes=includes)
    recorder.watched_object = t
    assert recorder.watched_object is t
    assert set(recorder.get_variables_list()) == set(expected)  # test lists regardless of order


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
def test_BaseRecorder_get_variables_list__excludes(AllTypesSystem, excludes, expected):
    s = AllTypesSystem('sub')
    t = AllTypesSystem('top')
    t.add_child(s)
    recorder = MockupRecorder(excludes=excludes)
    recorder.watched_object = t
    assert recorder.watched_object is t
    assert set(recorder.get_variables_list()) == set(expected)  # test lists regardless of order


def test_BaseRecorder__get_units(AllTypesSystem):
    recorder = MockupRecorder()
    assert recorder._get_units() == list()

    # Single system
    s = AllTypesSystem('test')
    recorder.watched_object = s
    assert len(recorder._get_units()) == len(recorder.get_variables_list())
    assert isinstance(recorder._get_units()[0], str)

    # Two levels system
    recorder = MockupRecorder()
    t = AllTypesSystem('top')
    t.add_child(s)
    recorder.watched_object = t
    assert len(recorder._get_units()) == len(recorder.get_variables_list())
    assert isinstance(recorder._get_units()[0], str)

    # Test includes
    recorder = MockupRecorder(includes='test.*')
    recorder.watched_object = t
    assert len(recorder._get_units()) == len(recorder.get_variables_list())
    assert isinstance(recorder._get_units()[0], str)

    recorder = MockupRecorder(includes='banana')
    recorder.watched_object = t
    assert len(recorder._get_units()) == 0

    # Test excludes
    recorder = MockupRecorder(excludes='test.*')
    recorder.watched_object = t
    assert len(recorder._get_units()) == len(recorder.get_variables_list())
    assert isinstance(recorder._get_units()[0], str)

    recorder = MockupRecorder(excludes='*')
    recorder.watched_object = t
    assert len(recorder._get_units()) == 0

    recorder = MockupRecorder(excludes='banana')
    recorder.watched_object = t
    assert len(recorder._get_units()) == len(recorder.get_variables_list())
    assert isinstance(recorder._get_units()[0], str)


def test_BaseRecorder_start():
    recorder = MockupRecorder()
    with pytest.raises(RuntimeError, match='A recorder should be watching a Driver'):
        recorder.start()


def test_BaseRecorder_record_state():
    with mock.patch("cosapp.core.signal.signal.inspect"):
        fake_callback = mock.Mock(spec=lambda **kwargs: None)
        fake_callback.return_value = None

        recorder = MockupRecorder()
        recorder.state_recorded.connect(Slot(fake_callback))
        recorder.record_state(time_ref='time', status='OK', error_code='000')

        fake_callback.assert_called_once_with(time_ref='time', status='OK', error_code='000')


def test_BaseRecorder_clear():
    with mock.patch("cosapp.core.signal.signal.inspect"):
        fake_callback = mock.Mock(spec=lambda **kwargs: None)
        fake_callback.return_value = None

        recorder = MockupRecorder()
        recorder.cleared.connect(Slot(fake_callback))
        recorder.record_state(time_ref='time', status='OK', error_code='000')
        recorder.clear()
        fake_callback.assert_called_once_with()
