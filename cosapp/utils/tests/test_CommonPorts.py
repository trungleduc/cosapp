import pytest
from cosapp.utils.naming import CommonPorts


@pytest.mark.parametrize("case", CommonPorts)
def test_CommonPorts_values(case):
    name = case.value
    assert isinstance(name, str)
    assert len(name) > 0
    assert name.strip() == name


def test_CommonPorts_names():
    assert set(CommonPorts.names()) == {
        'inwards',
        'outwards',
        'modevars_in',
        'modevars_out',
    }
