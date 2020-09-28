import pytest
from unittest import mock


@pytest.fixture
def mocked_slot_factory():
    def create_slot():
        slot = mock.Mock(spec=lambda **kwargs: None)
        slot.return_value = None
        return slot

    return create_slot


@pytest.fixture
def slot_function_factory():
    def slot_factory():
        def slot(**kwargs):
            pass

        return slot

    return slot_factory


@pytest.fixture
def slot_object_factory():
    class SlotObject:
        def __init__(self):
            self.called = False

        def slot(self, **kwargs):
            self.called = True

    return SlotObject


@pytest.fixture
def slot_factory(
    mocked_slot_factory,
    slot_object_factory,
    slot_function_factory,
):
    def factory(choice):
        choice = choice.lower()
        if choice.startswith('mock'):
            return mocked_slot_factory()
        if choice.startswith('func'):
            return slot_function_factory()
        if choice.startswith('obj'):
            return slot_object_factory()
        raise ValueError(f"argument {choice} not supported")
        
    return factory
