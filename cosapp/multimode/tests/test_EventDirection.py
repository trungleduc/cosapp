import pytest

from cosapp.multimode.zeroCrossing import EventDirection


@pytest.mark.parametrize("direction, prev, curr, expected", [
    # UP
    ("UP", 0, 1, True),
    ("UP", 0, 0, False),
    ("UP", -1, 0, True),
    ("UP", -1, 1, True),
    ("UP", 1, -1, False),
    ("UP", 0, -1, False),
    ("UP", 1, 0, False),
    # UPDOWN
    ("UPDOWN", 0, 1, True),
    ("UPDOWN", 0, 0, False),
    ("UPDOWN", -1, 0, True),
    ("UPDOWN", -1, 1, True),
    ("UPDOWN", 1, -1, True),
    ("UPDOWN", 0, -1, True),
    ("UPDOWN", 1, 0, True),
    # DOWN
    ("DOWN", 0, 1, False),
    ("DOWN", 0, 0, False),
    ("DOWN", -1, 0, False),
    ("DOWN", -1, 1, False),
    ("DOWN", 1, -1, True),
    ("DOWN", 0, -1, True),
    ("DOWN", 1, 0, True),
])
def test_EventDirection_zero_detected(direction, prev, curr, expected):
    eventdir = EventDirection[direction]
    assert eventdir.zero_detected(prev, curr) == expected
