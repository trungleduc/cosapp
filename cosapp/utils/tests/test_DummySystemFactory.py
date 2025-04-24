import pytest

from cosapp.base import Port, System
from cosapp.utils.testing import DummySystemFactory, get_args


class AbcPort(Port):
    def setup(self):
        self.add_variable('a', 1.0, unit='inch')
        self.add_variable('b', 1.0, unit='degF')
        self.add_variable('c', 1.0)

class XyzPort(Port):
    def setup(self):
        self.add_variable('x', unit='m')
        self.add_variable('y', unit='degC')
        self.add_variable('z', [0.1, 0.2, 0.3])


def get_name(obj) -> str:
    return obj.name


def test_DummySystemFactory():
    Dummy = DummySystemFactory(
        "Dummy",
        inputs=[
            get_args(AbcPort, 'abc_in'),
            get_args(XyzPort, 'xyz_in'),
        ],
        outputs=[
            get_args(AbcPort, 'abc_out'),
            get_args(XyzPort, 'xyz_out'),
        ],
        inwards=[
            get_args('h', 0.1, unit='m'),
        ],
        outwards=[
            get_args('b_ratio', 0.0),
        ],
        events=[
            get_args('kaboom', trigger='xyz_out.x > h')
        ],
        properties=[
            get_args('n', 12),
        ],
    )

    assert issubclass(Dummy, System)
    assert Dummy.__name__ == "Dummy"

    dummy = Dummy('dummy')
    assert set(dummy.inputs) == {
        'inwards',
        'modevars_in',
        'abc_in',
        'xyz_in',
    }
    assert set(dummy.outputs) == {
        'outwards',
        'modevars_out',
        'abc_out',
        'xyz_out',
    }
    assert set(map(get_name, dummy.events())) == {
        'kaboom',
    }
    assert dummy.properties == {'n': 12}


def test_DummySystemFactory_children():
    SystemA = DummySystemFactory(
        "SystemA",
        inputs=get_args(AbcPort, 'abc_in'),
        outputs=get_args(XyzPort, 'xyz_out'),
    )
    SystemB = DummySystemFactory(
        "SystemB",
        inputs=get_args(XyzPort, 'xyz_in'),
        outputs=get_args(AbcPort, 'abc_out'),
    )
    Top = DummySystemFactory(
        "Top",
        children=[
            get_args(SystemA('a'), pulling='abc_in'),
            get_args(SystemB('b'), pulling='abc_out'),
        ]
    )

    head = Top('head')
    assert list(head.exec_order) == ['a', 'b']
    assert set(head.inputs) == {
        'inwards',
        'modevars_in',
        'abc_in',
    }
    assert set(head.outputs) == {
        'outwards',
        'modevars_out',
        'abc_out',
    }
    assert set(map(get_name, head.all_connectors())) == {
        'abc_in -> a.abc_in',
        'b.abc_out -> abc_out',
    }
    
    head.connect(head.a.xyz_out, head.b.xyz_in)
    assert set(map(get_name, head.all_connectors())) == {
        'abc_in -> a.abc_in',
        'b.abc_out -> abc_out',
        'a.xyz_out -> b.xyz_in',
    }


def test_DummySystemFactory_wrong_settings():
    with pytest.warns(UserWarning, match=r"\['bar', 'foo'\] are not supported"):
        DummySystemFactory(
            "Dummy",
            inputs=[],
            outputs=[],
            inwards=[],
            outwards=[],
            events=[],
            properties=[],
            children=[],
            # Unsupported:
            foo=[],
            bar=[],
        )


def test_DummySystemFactory_base():
    BaseClass = DummySystemFactory(
        "SystemA",
        inputs=get_args(AbcPort, 'abc_in'),
        outputs=get_args(XyzPort, 'xyz_out'),
    )
    Extended = DummySystemFactory(
        "SystemB",
        base=BaseClass,
        inputs=get_args(XyzPort, 'xyz_in'),
        outputs=get_args(AbcPort, 'abc_out'),
    )

    assert issubclass(Extended, BaseClass)

    dummy = Extended('dummy')
    assert set(dummy.inputs) == {
        'inwards',
        'modevars_in',
        'abc_in',
        'xyz_in',
    }
    assert set(dummy.outputs) == {
        'outwards',
        'modevars_out',
        'abc_out',
        'xyz_out',
    }

    class Bogus:
        pass

    with pytest.raises(ValueError, match="`base` must be a type derived from `System`"):
        DummySystemFactory("Foo", base=Bogus)

    with pytest.raises(TypeError, match="`base` must be a type derived from `System`"):
        DummySystemFactory("Foo", base="string")
    