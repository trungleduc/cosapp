import pytest

from cosapp.recorders.recorder import BaseRecorder
from cosapp.recorders import DSVRecorder
from cosapp.tests.library.systems.vectors import AllTypesSystem


def test_DSVRecorder___init__default(tmp_path):
    frecorder = tmp_path / "file.csv"
    recorder = DSVRecorder(str(frecorder))

    assert recorder.includes == ["*"]
    assert recorder.excludes == list()
    assert recorder.section == ""
    assert recorder.precision == 9
    assert recorder.hold == False
    assert recorder._raw_output == False
    assert recorder.watched_object is None
    assert recorder._filepath == str(frecorder)
    assert recorder._delimiter == ","


@pytest.mark.parametrize(
    "args, kwargs, exception, match",
    [
        ((123,), dict(), TypeError, "'filepath' should be str"),
        (
            ("file.csv",),
            dict(delimiter="#"),
            ValueError,
            "Supported delimiters are .*; got '#'",
        ),
        (
            ("file.csv",),
            dict(includes=("q", 2)),
            TypeError,
            "'includes' must be a string, or a sequence of strings",
        ),
        (
            ("file.csv",),
            dict(includes=23),
            TypeError,
            "'includes' must be a string, or a sequence of strings",
        ),
        (
            ("file.csv",),
            dict(excludes=("q", 2)),
            TypeError,
            "'excludes' must be a string, or a sequence of strings",
        ),
        (
            ("file.csv",),
            dict(excludes=23),
            TypeError,
            "'excludes' must be a string, or a sequence of strings",
        ),
        (("file.csv",), dict(buffer="foo"), TypeError, "'buffer' should be bool"),
        (("file.csv",), dict(buffer=0), TypeError, "'buffer' should be bool"),
    ],
)
def test_DSVRecorder___init__error(args, kwargs, exception, match):
    with pytest.raises(exception, match=match):
        DSVRecorder(*args, **kwargs)


def test_DSVRecorder_start(tmp_path):
    frecorder = tmp_path / "records.csv"
    name = str(frecorder)

    rec = DSVRecorder(name)
    s = AllTypesSystem("test")
    rec.watched_object = s
    rec.start()

    assert frecorder.exists()
    content = frecorder.read_text().strip().split("\n")

    assert len(content) == 1
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    assert headers == [
        *BaseRecorder.SPECIALS,
        "a[0] [kg]",
        "a[1] [kg]",
        "a[2] [kg]",
        "b[0] [N]",
        "b[1] [N]",
        "b[2] [N]",
        "c [m]",
        "d[0] [-]",
        "d[1] [-]",
        "e [-]",
        "in_.x[0] [-]",
        "in_.x[1] [-]",
        "in_.x[2] [-]",
        "out.x[0] [-]",
        "out.x[1] [-]",
        "out.x[2] [-]",
    ]

    rec = DSVRecorder(name, raw_output=True)
    rec.watched_object = s
    rec.start()

    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 1
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    assert headers == [
        *BaseRecorder.SPECIALS,
        "a[0]",
        "a[1]",
        "a[2]",
        "b[0]",
        "b[1]",
        "b[2]",
        "c",
        "d[0]",
        "d[1]",
        "e",
        "in_.x[0]",
        "in_.x[1]",
        "in_.x[2]",
        "out.x[0]",
        "out.x[1]",
        "out.x[2]",
    ]

    rec.record_state(0)
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 2
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    assert headers == [
        *BaseRecorder.SPECIALS,
        "a[0]",
        "a[1]",
        "a[2]",
        "b[0]",
        "b[1]",
        "b[2]",
        "c",
        "d[0]",
        "d[1]",
        "e",
        "in_.x[0]",
        "in_.x[1]",
        "in_.x[2]",
        "out.x[0]",
        "out.x[1]",
        "out.x[2]",
    ]
    data = content[1].strip().split(rec._delimiter)
    assert len(data) == (16 + len(BaseRecorder.SPECIALS))

    rec.record_state(1)
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 3
    data = content[2].strip().split(rec._delimiter)
    assert len(data) == 16 + len(BaseRecorder.SPECIALS)

    rec.start()
    rec.record_state(0)
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 2
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    assert headers == [
        *BaseRecorder.SPECIALS,
        "a[0]",
        "a[1]",
        "a[2]",
        "b[0]",
        "b[1]",
        "b[2]",
        "c",
        "d[0]",
        "d[1]",
        "e",
        "in_.x[0]",
        "in_.x[1]",
        "in_.x[2]",
        "out.x[0]",
        "out.x[1]",
        "out.x[2]",
    ]

    data = content[1].strip().split(rec._delimiter)
    assert len(data) == (16 + len(BaseRecorder.SPECIALS))

    # Test hold
    rec.hold = True
    rec.start()
    rec.record_state(1)
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 3
    data = content[2].strip().split(rec._delimiter)
    assert len(data) == (16 + len(BaseRecorder.SPECIALS))


def test_DSVRecorder_record_iteration(tmp_path):
    frecorder = tmp_path / "recorder.csv"
    name = str(frecorder)

    s = AllTypesSystem("test")
    s.run_once()

    rec = DSVRecorder(name, raw_output=True)
    rec.watched_object = s

    rec.record_state(0)
    precision = 9
    line = ["", "", "0", "0"] + ["{0:.{1}e}".format(1.0, precision)] * 3
    line.extend(["{0:.{1}e}".format(0.0, precision)] * 3)
    line.append("{0:.{1}e}".format(23, precision))
    line.append("{0:.{1}e}".format(23, precision))
    line.append("sammy")
    line.append("sammy")
    line.append("{0:.{1}e}".format(1.0, precision))
    line.append("{0:.{1}e}".format(2.0, precision))
    line.append("{0:.{1}e}".format(3.0, precision))
    line.append("{0:.{1}e}".format(1.0, precision))
    line.append("{0:.{1}e}".format(2.0, precision))
    line.append("{0:.{1}e}".format(3.0, precision))

    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 1
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    assert headers == line

    rec.record_state(1)
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 2
    data = content[1].strip().split(rec._delimiter)
    assert len(data) == (16 + len(BaseRecorder.SPECIALS))


def test_DSVRecorder_record_precision(tmp_path):
    frecorder = tmp_path / "recorder.csv"
    name = str(frecorder)
    s = AllTypesSystem("test")
    s.run_once()

    rec = DSVRecorder(name, raw_output=True, precision=4)
    rec.watched_object = s

    rec.record_state(0)
    precision = 4
    line = ["", "", "0", "0"] + ["{0:.{1}e}".format(1.0, precision)] * 3
    line.extend(["{0:.{1}e}".format(0.0, precision)] * 3)
    line.append("{0:.{1}e}".format(23, precision))
    line.append("{0:.{1}e}".format(23, precision))
    line.append("sammy")
    line.append("sammy")
    line.append("{0:.{1}e}".format(1.0, precision))
    line.append("{0:.{1}e}".format(2.0, precision))
    line.append("{0:.{1}e}".format(3.0, precision))
    line.append("{0:.{1}e}".format(1.0, precision))
    line.append("{0:.{1}e}".format(2.0, precision))
    line.append("{0:.{1}e}".format(3.0, precision))

    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 1
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    assert headers == line


def test_DSVRecorder_exit(tmp_path):
    frecorder = tmp_path / "recorder.csv"
    name = str(frecorder)
    s = AllTypesSystem("test")

    rec = DSVRecorder(name, raw_output=True)
    rec.watched_object = s

    rec.start()
    rec.record_state(0)
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 2
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    data = content[1].strip().split(rec._delimiter)
    assert len(data) == (16 + len(BaseRecorder.SPECIALS))

    rec.exit()
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 2
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    data = content[1].strip().split(rec._delimiter)
    assert len(data) == (16 + len(BaseRecorder.SPECIALS))

    # Test buffer
    rec = DSVRecorder(name, raw_output=True, buffer=True)
    rec.watched_object = s

    rec.start()
    rec.record_state(0)
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 1
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))

    rec.exit()
    content = frecorder.read_text().strip().split("\n")
    assert len(content) == 2
    headers = content[0].strip().split(rec._delimiter)
    assert len(headers) == (16 + len(BaseRecorder.SPECIALS))
    data = content[1].strip().split(rec._delimiter)
    assert len(data) == (16 + len(BaseRecorder.SPECIALS))
