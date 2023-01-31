import pytest

from cosapp.utils import partition


@pytest.mark.parametrize("collection, predicate, expected", [
    (
        range(10), lambda n: n % 2 == 0,
        dict(
            yays = [0, 2, 4, 6, 8],
            nays = [1, 3, 5, 7, 9],
        ),
    ),
    (
        range(10), lambda n: n > 6,
        dict(
            yays = [7, 8, 9],
            nays = [0, 1, 2, 3, 4, 5, 6],
        ),
    ),
    (
        "CoSApp", lambda letter: letter.isupper(),
        dict(
            yays = list("CSA"),
            nays = list("opp"),
        ),
    ),
])
def test_partition(collection, predicate, expected):
    yays, nays = partition(collection, predicate)
    assert yays == expected['yays']
    assert nays == expected['nays']
