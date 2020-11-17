import pytest
from cosapp.systems import System
from cosapp.tests.library.ports import XPort
from cosapp.systems import FloatKrigingSurrogate, LinearNearestNeighbor
from cosapp.systems.surrogateclass import (
    SurrogateClass, SurrogateClassState, flatten,
    get_dependant_connections,
)
from cosapp.drivers import RunOnce, NonLinearSolver, RunSingleCase, EulerExplicit
import numpy
import pandas
import itertools
from unittest.mock import patch, Mock, DEFAULT
from cosapp.tests.library.systems.basicalgebra import (
    Sys_Provider_1Eq_2Mult_Getter,
    Sys_PME2MUG_G_1E,
    Sys_Sum3_2Eq_2Mult,
    Sys_Unknown_1Eq_2Mult_Getter,
    Sys_Lopped_Div_Int_Div,
)
from cosapp.tests.library.systems.vectors import Strait1dLine, Splitter1d
from scipy.linalg import LinAlgWarning
from cosapp.utils.testing import get_args


"""
meta model can be used on:
- systems without any driver on one or more of their children
- systems without any driver on themself
- systems with NLS (NonLinearSolver) on themself
- systems with NLS on one or more of their children

(#TODO ?) meta model can not be used on:
- systems with time driver on themself
- systems with time driver on one or more of their children

=> Brain storm to have about the kind of data we want to record if  we implement training with time driver on system and how to manage it?

Once the training is done, you can add any type of driver (NLS, EulerExplicit, RungeKutta)
"""


list_for_get_dependant_connections = [
    (Sys_Provider_1Eq_2Mult_Getter('p1e2mg').Eq_2Mult, {
        'x_out.x': [], 'u_out.x': [], 
        'Mult_by_2_1.x_out.x': [], 'Mult_by_2_1.x_in.x': [], 
        'Mult_by_2_2.x_out.x': [], 'Mult_by_2_2.x_in.x': []
    }),
    (Sys_Provider_1Eq_2Mult_Getter('p1e2mg'), {
        'Provider.x_out.x': [], 'Provider.u_out.x': [], 
        'Eq_2Mult.x_out.x': [], 'Eq_2Mult.u_out.x': [], 
        'Eq_2Mult.x_in.x': [], 'Eq_2Mult.u_in.x': [], 
        'Eq_2Mult.Mult_by_2_1.x_out.x': [], 'Eq_2Mult.Mult_by_2_1.x_in.x': [], 
        'Eq_2Mult.Mult_by_2_2.x_out.x': [], 'Eq_2Mult.Mult_by_2_2.x_in.x': [], 
        'Get2D.x_out.x': [], 'Get2D.u_out.x': [], 
        'Get2D.x_in.x': [], 'Get2D.u_in.x': []
    }),
    (Sys_PME2MUG_G_1E("pme"), {
        'PMEMUG.x_out.x': [], 'PMEMUG.u_out.x': [], 
        'PMEMUG.SMXI.x_out.x': [], 'PMEMUG.SMXI.x_in.x': [], 
        'PMEMUG.Get2D.x_out.x': [], 'PMEMUG.Get2D.u_out.x': [], 
        'PMEMUG.Get2D.x_in.x': [], 'PMEMUG.Get2D.u_in.x': [], 
        'PMEMUG.S1E2M.x_out.x': [], 'PMEMUG.S1E2M.u_out.x': [], 
        'PMEMUG.S1E2M.u_in.x': [], 'PMEMUG.S1E2M.x_in.x': [], 
        'PMEMUG.S1E2M.Mult_by_2_1.x_out.x': [], 'PMEMUG.S1E2M.Mult_by_2_1.x_in.x': [], 
        'PMEMUG.S1E2M.Mult_by_2_2.x_out.x': [], 'PMEMUG.S1E2M.Mult_by_2_2.x_in.x': [], 
        'G2DEq.x_out.x': [], 'G2DEq.u_out.x': [], 
        'G2DEq.x_in.x': [], 'G2DEq.u_in.x': []
    }),
    (Sys_Sum3_2Eq_2Mult("ss3e2m2"), {
        'u_out.x': [], 'x_out.x': [], 'Mult_by_2_1.x_out.x': [], 
        'Mult_by_2_1.x_in.x': [], 'Mult_by_2_2.x_out.x': [], 
        'Mult_by_2_2.x_in.x': [], 'Basic_Eq.x_in.x': []
    }),
    (Sys_PME2MUG_G_1E("pme").PMEMUG, {
        'x_out.x': [], 'u_out.x': [], 
        'SMXI.x_out.x': [], 'SMXI.x_in.x': [], 
        'Get2D.x_out.x': [], 'Get2D.u_out.x': [], 
        'Get2D.x_in.x': [], 'Get2D.u_in.x': [], 
        'S1E2M.x_out.x': [], 'S1E2M.u_out.x': [], 
        'S1E2M.u_in.x': [], 'S1E2M.x_in.x': [], 
        'S1E2M.Mult_by_2_1.x_out.x': [], 'S1E2M.Mult_by_2_1.x_in.x': [], 
        'S1E2M.Mult_by_2_2.x_out.x': [], 'S1E2M.Mult_by_2_2.x_in.x': []
    }),
    (Sys_Unknown_1Eq_2Mult_Getter("sumg"), {
        'Eq_2Mult.Basic_Eq.x_in.x': [], 'Eq_2Mult.Mult_by_2_1.x_in.x': [],
        'Eq_2Mult.Mult_by_2_1.x_out.x': [],'Eq_2Mult.Mult_by_2_2.x_in.x': [],
        'Eq_2Mult.Mult_by_2_2.x_out.x': [],'Eq_2Mult.u_in.x': [],
        'Eq_2Mult.u_out.x': [],'Eq_2Mult.x_in.x': [],'Eq_2Mult.x_out.x': [],
        'Get2D.u_in.x': [],'Get2D.u_out.x': [],'Get2D.x_in.x': [],
        'Get2D.x_out.x': [],'Provider.u_out.x': [],'Provider.x_out.x': []
    }),
    (Sys_Unknown_1Eq_2Mult_Getter("sumg").Provider, {'u_out.x': [], 'x_out.x': []}),
]


class A(System):
    def setup(self):
        self.add_input(XPort, 'a_in')
        self.add_child(C('c'))

class B(System):
    def setup(self):
        self.add_inward('u')
        self.add_unknown('u')

class C(System):
    def setup(self):
        self.add_input(XPort, 'j')
        self.add_transient('h', der = 'j.x')

class E(System):
    def setup(self):
        self.add_child(A('a'))
        self.add_child(B('b'))


@pytest.fixture(scope="function")
def no_output_sys():
    return E('no_output_sys')


@pytest.fixture(scope="function")
def warning_unknowns_transients_sys():
    class F(System):
        def setup(self):
            self.add_child(E('e'))
            self.add_outward('h')
    
    return F('warning_UT_Sys')


@pytest.fixture(scope="function")
def cubic_DoE():
    def factory(dic):
        return pandas.DataFrame(
            list(itertools.product(*dic.values())),
            columns = [key for key in dic]
        )
    return factory


@pytest.fixture(scope="function")
def list_of_subsystems3():
    return [
        (
            Sys_Provider_1Eq_2Mult_Getter('p1e2mg').Eq_2Mult, 
            pytest.approx(14.885860116443425, rel=1.e-10), 
            pytest.approx(13.385860116443887, rel=1.e-10),
            pytest.approx(14.5, rel=0.),
            pytest.approx(13.0, rel=0.),
        ),
        (
            Sys_PME2MUG_G_1E("pme").PMEMUG,
            pytest.approx(111.19501618681366, rel=1.e-6),
            pytest.approx(110.00000000218027, rel=1.e-6),
            pytest.approx(111.33329999996438, rel=0.),
            pytest.approx(109.99999999996439, rel=0.),
        ),
        (
            Sys_Unknown_1Eq_2Mult_Getter("sumg").Provider,
            pytest.approx(15.0, rel=1.e-5),
            pytest.approx(12.5, rel=1.e-9),
            pytest.approx(15.0, rel=0),
            pytest.approx(12.5),
        ),
    ]


@pytest.fixture(scope="function")
def list_of_subsystems():
    return [
        Sys_Provider_1Eq_2Mult_Getter('p1e2mg').Eq_2Mult,
        Sys_Sum3_2Eq_2Mult("ss3e2m2"),
        Sys_PME2MUG_G_1E("pme").PMEMUG,
        Sys_Unknown_1Eq_2Mult_Getter("sumg").Provider,
    ]


@pytest.fixture(scope="function")
def unknown_sys_training_data(cubic_DoE):
    dic = {'PMEMUG.x_in.x': numpy.linspace(-3,3,40)}
    return cubic_DoE(dic)


@pytest.fixture(scope="function")
def vectorsys():
    sys = Strait1dLine("sys")
    return sys


@pytest.fixture(scope="function")
def vectorsplit():
    sys = Splitter1d("sys")
    return sys


@pytest.fixture(scope="function")
def data_vectorsys(cubic_DoE):
    dic = {
        'a': list((k + 1) * numpy.ones(3) for k in range(3)),
        'in_.x':list(k * 0.5 * numpy.ones(3) for k in range(3)),
    }
    return cubic_DoE(dic)


@pytest.fixture(scope="function")
def data_vectorsys_HF(cubic_DoE):
    dic = {
        'a': list((k + 1) * numpy.ones(3) for k in range(5)),
        'in_.x': list(k * 0.5 * numpy.ones(3) for k in range(5)),
    }
    return cubic_DoE(dic)


@pytest.fixture(scope="function")
def data_vectorsplit_HF(cubic_DoE):
    dic = {
        'in_.x': list(k * 0.15 * numpy.ones(3) for k in range(15)),
    }
    return cubic_DoE(dic)


@pytest.fixture(scope="function")
def list_of_systems():
    return [
        (Sys_Provider_1Eq_2Mult_Getter('p1e2mg').Eq_2Mult),
        (Sys_Provider_1Eq_2Mult_Getter('p1e2mg')),
        (Sys_PME2MUG_G_1E("pme")),
        (Sys_Sum3_2Eq_2Mult("ss3e2m2")),
        (Sys_PME2MUG_G_1E("pme").PMEMUG),
        (Sys_Unknown_1Eq_2Mult_Getter("sumg")),
        (Sys_Unknown_1Eq_2Mult_Getter("sumg").Provider),
    ]


@pytest.fixture(scope = "function")
def nude_sumg():
    return Sys_Unknown_1Eq_2Mult_Getter("sumg")


@pytest.fixture(scope="function")
def pme():
    #Surrogated System
    pme = Sys_PME2MUG_G_1E("pme")
    nls1 = pme.add_driver(NonLinearSolver("nls1"))
    rsc1 = nls1.add_child(RunSingleCase('rsc1'))
    return pme


@pytest.fixture(scope="function")
def surrogated_pme_LNN(unknown_sys_training_data):
    #Surrogated System
    pme = Sys_PME2MUG_G_1E("pme")
    nls1 = pme.add_driver(NonLinearSolver("nls1"))
    rsc1 = nls1.add_child(RunSingleCase('rsc1'))
    meta = pme.make_surrogate(unknown_sys_training_data, LinearNearestNeighbor)
    return pme

@pytest.fixture(scope="function")
def training_data_pme(cubic_DoE):
    dic = {'x_in.x': [0.5, 1., 1.5], 'u_in.x': [0.5, 1., 1.3333], 'SMXI.m.x' : numpy.linspace(23., 28., 6)}
    return cubic_DoE(dic)


@pytest.fixture(scope="function")
def p1e2mg():
    p1e2mg = Sys_Provider_1Eq_2Mult_Getter('p1e2mg')
    return p1e2mg


@pytest.fixture(scope="function")
def metadico_out():
    metadico_out = {'x_out.x': [], 'u_out.x': [], 'Mult_by_2_1.x_out.x': [], 'Mult_by_2_1.x_in.x': [], 'Mult_by_2_2.x_out.x': [], 'Mult_by_2_2.x_in.x': []}
    return metadico_out


@pytest.fixture(scope="function")
def sumg():
    sumg = Sys_Unknown_1Eq_2Mult_Getter("sumg")
    nls = sumg.add_driver(NonLinearSolver('nls'))
    rsc = nls.add_child(RunSingleCase('rsc'))
    return sumg


@pytest.fixture(scope="function")
def training_data3(cubic_DoE):
    dic = {'x_in.x': numpy.linspace(0., 3., 3), 'u_in.x': numpy.linspace(0., 3., 3)}
    return cubic_DoE(dic)


@pytest.fixture(scope="function")
def training_data15(cubic_DoE):
    dic = {'x_in.x': numpy.linspace(0., 15., 15), 'u_in.x': numpy.linspace(0., 15., 15)}
    return cubic_DoE(dic)


@pytest.fixture(scope="function")
def data_in(cubic_DoE):
    dic = {'x_in.x' : [2., 2.333, 2.666, 3.], 'u_in.x' : [1., 1.3333, 1.6666, 2.]}
    return cubic_DoE(dic)


@pytest.fixture(scope="function")
def case_factory(training_data_pme, training_data3, training_data15):
    def factory(name):
        if name == "Eq_2Mult":
            main = Sys_Provider_1Eq_2Mult_Getter('p1e2mg')
            target = main.Eq_2Mult
            data = training_data3
        elif name == "PMEMUG":
            main = Sys_PME2MUG_G_1E("pme")
            target = main.PMEMUG
            data = training_data_pme
        elif name == "Provider":
            main = Sys_Unknown_1Eq_2Mult_Getter("sumg")
            target = main.Provider
            data = training_data15
    
        target.make_surrogate(data, FloatKrigingSurrogate)
        nls = main.add_driver(NonLinearSolver('nls'))
        rsc = nls.add_child(RunSingleCase('rsc'))
        return main, target

    return factory


@pytest.mark.parametrize("L_to_flat, expected_list", [
    ([[1,2,3],[4],[5,6,7,8],[9]], [1,2,3,4,5,6,7,8,9]),
    ([[1,2,3],[4],[5,6,7,8],[9,[10, 11]]], [1,2,3,4,5,6,7,8,9,10,11]),
])
def test_flatten(L_to_flat, expected_list):
    meta = SurrogateClass(None, None, None)
    assert list(flatten(L_to_flat)) == expected_list


@pytest.mark.parametrize("state, expected", [
    (
        {
            "Eq_2Mult.Mult_by_2_2.x_out.x": 147,
            "Eq_2Mult.x_out.x": 123.,
            "Eq_2Mult.u_out.x": 456.,
        },
        {
            'x_out.x': [123.0],
            'u_out.x': [456.0],
            'Mult_by_2_1.x_out.x': [1],
            'Mult_by_2_1.x_in.x': [1],
            'Mult_by_2_2.x_out.x': [147],
            'Mult_by_2_2.x_in.x': [1],
        }
    ),
    (
        {
            "Eq_2Mult.Mult_by_2_2.x_out.x": 147,
            "Eq_2Mult.x_out.x": 123.,
            "Eq_2Mult.u_out.x": 456.,
        },
        {
            'x_out.x': [123.0],
            'u_out.x': [456.0],
            'Mult_by_2_1.x_out.x': [1],
            'Mult_by_2_1.x_in.x': [1],
            'Mult_by_2_2.x_out.x': [147],
            'Mult_by_2_2.x_in.x': [1],
        }
    ),
])
def test_SurrogateClass_update_doe_out(p1e2mg, metadico_out, state, expected):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, None, FloatKrigingSurrogate)
    for var, value in state.items():
        p1e2mg[var] = value
    meta._SurrogateClass__update_doe_out()
    assert meta.state.doe_out == expected


@pytest.mark.parametrize("expected",[
    ({'x_in.x' : 25, 'u_in.x' : 50}),
    ({'x_in.x' : 0., 'u_in.x' : 20}),
    ({'x_in.x' : -100, 'u_in.x' : 15}),
    ({'x_in.x' : 2512/423, 'u_in.x' : -3.7586}),
])
def test_SurrogateClass_set_owner_inputs(p1e2mg, expected):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, None, None)
    for val in expected:
        p1e2mg.Eq_2Mult[val] = -numpy.inf
    meta._SurrogateClass__set_owner_inputs(expected)
    for key, val in expected.items():
        assert meta.owner[key] == val


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_format_inputs(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    formated_inputs = meta._SurrogateClass__format_inputs()
    expected_formated_inputs = numpy.array([[2., 1.],[2., 1.3333],
                                            [2., 1.6666], [2., 2.],
                                            [2.333, 1.],[2.333, 1.3333],
                                            [2.333, 1.6666],[2.333, 2.],
                                            [2.666, 1.],[2.666, 1.3333],
                                            [2.666, 1.6666],[2.666, 2.],
                                            [3., 1.],[3., 1.3333],
                                            [3., 1.6666],[3., 2.]]
    )
    assert formated_inputs == pytest.approx(expected_formated_inputs, rel=1e-9)
    assert numpy.array_equal(formated_inputs, expected_formated_inputs)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_format_inputs_arrays(vectorsys, data_vectorsys):
    meta = SurrogateClass(vectorsys, data_vectorsys, FloatKrigingSurrogate)
    formated_inputs = meta._SurrogateClass__format_inputs()
    expected_formated_inputs = numpy.array([
        [1. , 1. , 1. , 0. , 0. , 0. ],[1. , 1. , 1. , 0.5, 0.5, 0.5],[1. , 1. , 1. , 1. , 1. , 1. ],
        [2. , 2. , 2. , 0. , 0. , 0. ],[2. , 2. , 2. , 0.5, 0.5, 0.5],[2. , 2. , 2. , 1. , 1. , 1. ],
        [3. , 3. , 3. , 0. , 0. , 0. ],[3. , 3. , 3. , 0.5, 0.5, 0.5],[3. , 3. , 3. , 1. , 1. , 1. ]
    ])
    assert formated_inputs == pytest.approx(expected_formated_inputs, rel=1e-9)
    assert numpy.array_equal(formated_inputs, expected_formated_inputs)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_format_outputs(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    formated_outputs = meta._SurrogateClass__format_outputs()
    expected_format_outputs = numpy.array([
        [11., 12., 2., 4., 4., 8.],
        [11., 12.3333, 2., 4., 4., 8.],
        [11., 12.6666, 2., 4., 4., 8.],
        [11., 13., 2., 4., 4., 8.],
        [12.332, 13.332 , 2.333, 4.666, 4.666, 9.332],
        [12.332, 13.6653, 2.333, 4.666, 4.666, 9.332],
        [12.332, 13.9986, 2.333, 4.666, 4.666, 9.332],
        [12.332, 14.332 , 2.333, 4.666, 4.666, 9.332],
        [13.664, 14.664 , 2.666, 5.332, 5.332, 10.664],
        [13.664, 14.9973, 2.666, 5.332, 5.332, 10.664],
        [13.664, 15.3306, 2.666, 5.332, 5.332, 10.664],
        [13.664, 15.664 , 2.666, 5.332, 5.332, 10.664],
        [15., 16., 3., 6., 6., 12.],
        [15., 16.3333, 3., 6., 6., 12.],
        [15., 16.6666, 3., 6., 6., 12.],
        [15., 17., 3., 6., 6., 12.],
    ])
    assert formated_outputs == pytest.approx(expected_format_outputs)


def test_SurrogateClass_format_outputs_arrays(vectorsys, data_vectorsys):
    meta = SurrogateClass(vectorsys, data_vectorsys, FloatKrigingSurrogate)
    formated_outputs = meta._SurrogateClass__format_outputs()
    expected_formated_outputs = numpy.array([
        [0. , 0. , 0. ],[0.5, 0.5, 0.5],[1. , 1. , 1. ],
        [0. , 0. , 0. ],[1. , 1. , 1. ],[2. , 2. , 2. ],
        [0. , 0. , 0. ],[1.5, 1.5, 1.5],[3. , 3. , 3. ]
    ])
    assert formated_outputs == pytest.approx(expected_formated_outputs, rel=1e-9)
    assert numpy.array_equal(formated_outputs, expected_formated_outputs)


def test_SurrogateClass_train_model(p1e2mg):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, None, FloatKrigingSurrogate)
    modelOK = FloatKrigingSurrogate()
    x_in = numpy.array([[1.,2.,3.,4.,5.]]).transpose()
    x_out = numpy.array([[2.,4.,6.,8.,10.]]).transpose()
    modelOK.train(x_in, x_out)
    meta._SurrogateClass__train_model(x_in, x_out)
    for k in [1.5, 2.5, 3.5, 4.5]:
        assert modelOK.predict(k) == meta.state.model.predict(k)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_set_and_execute_check_set_inputs_execution(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    with patch('cosapp.systems.surrogateclass.SurrogateClass._SurrogateClass__set_owner_inputs') as mocked_method:
        meta._SurrogateClass__set_and_execute()
        assert mocked_method.call_count == 16


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_set_and_execute_check_get_outputs_execution(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    with patch('cosapp.systems.surrogateclass.SurrogateClass._SurrogateClass__update_doe_out') as mocked_method:
        meta._SurrogateClass__set_and_execute()
        assert mocked_method.call_count == 16


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_prepare_and_train_in_init(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    expected_doe_out = {
        'x_out.x': [11.0, 11.0, 11.0, 11.0, 12.332, 12.332, 12.332, 12.332, 13.664, 13.664, 13.664, 13.664, 15.0, 15.0, 15.0, 15.0],
        'u_out.x': [12.0, 12.3333, 12.6666, 13.0, 13.332, 13.6653, 13.998600000000001, 14.332, 14.664, 14.9973, 15.3306, 15.664, 16.0, 16.3333, 16.6666, 17.0],
        'Mult_by_2_1.x_out.x': [4.0, 4.0, 4.0, 4.0, 4.666, 4.666, 4.666, 4.666, 5.332, 5.332, 5.332, 5.332, 6.0, 6.0, 6.0, 6.0],
        'Mult_by_2_1.x_in.x': [2.0, 2.0, 2.0, 2.0, 2.333, 2.333, 2.333, 2.333, 2.666, 2.666, 2.666, 2.666, 3.0, 3.0, 3.0, 3.0],
        'Mult_by_2_2.x_out.x': [8.0, 8.0, 8.0, 8.0, 9.332, 9.332, 9.332, 9.332, 10.664, 10.664, 10.664, 10.664, 12.0, 12.0, 12.0, 12.0],
        'Mult_by_2_2.x_in.x': [4.0, 4.0, 4.0, 4.0, 4.666, 4.666, 4.666, 4.666, 5.332, 5.332, 5.332, 5.332, 6.0, 6.0, 6.0, 6.0]
    }
    assert meta.state.doe_out == expected_doe_out


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_prepare_and_train_check_functions_execution(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    with patch.multiple("cosapp.systems.surrogateclass.SurrogateClass",
        _SurrogateClass__format_outputs = DEFAULT,
        _SurrogateClass__format_inputs = DEFAULT,
        _SurrogateClass__set_and_execute = DEFAULT,
        _SurrogateClass__train_model = DEFAULT,
        ) as mocked_methods:
        meta._SurrogateClass__prepare_and_train()
        mocked_methods['_SurrogateClass__format_outputs'].assert_called_once()
        mocked_methods['_SurrogateClass__format_inputs'].assert_called_once()
        mocked_methods['_SurrogateClass__set_and_execute'].assert_called_once()
        mocked_methods['_SurrogateClass__train_model'].assert_called_once()


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_prepare_and_train_in_init_predict(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    expected_doe_out = {
        'x_out.x': [11.0, 11.0, 11.0, 11.0, 12.332, 12.332, 12.332, 12.332, 13.664, 13.664, 13.664, 13.664, 15.0, 15.0, 15.0, 15.0],
        'u_out.x': [12.0, 12.3333, 12.6666, 13.0, 13.332, 13.6653, 13.998600000000001, 14.332, 14.664, 14.9973, 15.3306, 15.664, 16.0, 16.3333, 16.6666, 17.0],
        'Mult_by_2_1.x_out.x': [4.0, 4.0, 4.0, 4.0, 4.666, 4.666, 4.666, 4.666, 5.332, 5.332, 5.332, 5.332, 6.0, 6.0, 6.0, 6.0],
        'Mult_by_2_1.x_in.x': [2.0, 2.0, 2.0, 2.0, 2.333, 2.333, 2.333, 2.333, 2.666, 2.666, 2.666, 2.666, 3.0, 3.0, 3.0, 3.0],
        'Mult_by_2_2.x_out.x': [8.0, 8.0, 8.0, 8.0, 9.332, 9.332, 9.332, 9.332, 10.664, 10.664, 10.664, 10.664, 12.0, 12.0, 12.0, 12.0],
        'Mult_by_2_2.x_in.x': [4.0, 4.0, 4.0, 4.0, 4.666, 4.666, 4.666, 4.666, 5.332, 5.332, 5.332, 5.332, 6.0, 6.0, 6.0, 6.0]
    }
    assert meta.state.doe_out == expected_doe_out
    assert meta.state.model.predict(numpy.array([2.5,1.5])) == pytest.approx([13.00059934, 14.50059842, 2.50014984, 5.00029967, 5.00029967, 10.00059934], rel=1e-7)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
@pytest.mark.parametrize("expected",[
    ({'x_in.x' : 25, 'u_in.x' : 50}),
    ({'x_in.x' : 0., 'u_in.x' : 20}),
    ({'x_in.x' : -100, 'u_in.x' : 15}),
    ({'x_in.x' : 2512/423, 'u_in.x' : -3.7586}),
])
def test_SurrogateClass_get_current_inputs_result(p1e2mg,data_in, expected):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    p1e2mg.Eq_2Mult["x_in.x"] = expected["x_in.x"]
    p1e2mg.Eq_2Mult["u_in.x"] = expected["u_in.x"]
    assert p1e2mg.Eq_2Mult.x_in.x == expected["x_in.x"]
    assert p1e2mg.Eq_2Mult.u_in.x == expected["u_in.x"]
    assert numpy.array_equal(meta._SurrogateClass__get_current_inputs(), [expected["x_in.x"], expected["u_in.x"]])


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_set_current_outputs_result(p1e2mg,data_in):
    np_outputs = numpy.array([5., 2., 10., 3., 7., 15.])
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    meta._SurrogateClass__set_current_outputs(np_outputs)
    assert p1e2mg.Eq_2Mult.x_out.x == 5.0
    assert p1e2mg.Eq_2Mult.u_out.x ==  2.0
    assert p1e2mg.Eq_2Mult.Mult_by_2_1.x_out.x == 3.0
    assert p1e2mg.Eq_2Mult.Mult_by_2_1.x_in.x ==  10.0
    assert p1e2mg.Eq_2Mult.Mult_by_2_2.x_out.x == 15.0
    assert p1e2mg.Eq_2Mult.Mult_by_2_2.x_in.x ==  7.0


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_compute_execution(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    with patch.multiple("cosapp.systems.surrogateclass.SurrogateClass", _SurrogateClass__set_current_outputs = DEFAULT,_SurrogateClass__get_current_inputs = DEFAULT) as mocked_methods:
        with patch('cosapp.systems.surrogate_models.kriging.FloatKrigingSurrogate.predict') as mocked_predict:
            meta.compute()
            mocked_methods["_SurrogateClass__set_current_outputs"].assert_called_once()
            mocked_methods["_SurrogateClass__get_current_inputs"].assert_called_once()
            mocked_predict.assert_called_once()


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_compute_result(p1e2mg,data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    p1e2mg.run_drivers()
    meta.compute()
    assert p1e2mg.Eq_2Mult.x_out.x == pytest.approx(13., rel= 1.e-3)
    assert p1e2mg.Eq_2Mult.u_out.x == pytest.approx(14.5, rel=1.e-3)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_global_working_0(pme, training_data_pme):
    #Surrogated System
    pme.PMEMUG.make_surrogate(training_data_pme, FloatKrigingSurrogate)
    pme.PMEMUG.x_in.x = 1.
    pme.PMEMUG.u_in.x = 1.
    pme.run_drivers()
    assert pme.PMEMUG.SMXI.m.x == pytest.approx(26.75, rel=1e-4)
    assert pme.PMEMUG.x_out.x == pytest.approx(110., rel=1e-4)
    assert pme.PMEMUG.u_out.x == pytest.approx(111., rel=1e-4)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_init1(p1e2mg, data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    assert meta is not None
    assert meta.owner is p1e2mg.Eq_2Mult
    expected_doe_in = pandas.DataFrame(numpy.array([
        [2.0, 1.0], [2.0, 1.3333], 
        [2.0, 1.6666], [2.0, 2.0], 
        [2.333, 1.0], [2.333, 1.3333], 
        [2.333, 1.6666], [2.333, 2.0], 
        [2.666, 1.0], [2.666, 1.3333], 
        [2.666, 1.6666], [2.666, 2.0], 
        [3.0, 1.0], [3.0, 1.3333], 
        [3.0, 1.6666], [3.0, 2.0]]), 
        columns = ['x_in.x', 'u_in.x']
    )
    expected_doe_out = {
        'x_out.x': [11.0, 11.0, 11.0, 11.0, 12.332, 12.332, 12.332, 12.332, 13.664, 13.664, 13.664, 13.664, 15.0, 15.0, 15.0, 15.0],
        'u_out.x': [12.0, 12.3333, 12.6666, 13.0, 13.332, 13.6653, 13.998600000000001, 14.332, 14.664, 14.9973, 15.3306, 15.664, 16.0, 16.3333, 16.6666, 17.0],
        'Mult_by_2_1.x_out.x': [4.0, 4.0, 4.0, 4.0, 4.666, 4.666, 4.666, 4.666, 5.332, 5.332, 5.332, 5.332, 6.0, 6.0, 6.0, 6.0],
        'Mult_by_2_1.x_in.x': [2.0, 2.0, 2.0, 2.0, 2.333, 2.333, 2.333, 2.333, 2.666, 2.666, 2.666, 2.666, 3.0, 3.0, 3.0, 3.0],
        'Mult_by_2_2.x_out.x': [8.0, 8.0, 8.0, 8.0, 9.332, 9.332, 9.332, 9.332, 10.664, 10.664, 10.664, 10.664, 12.0, 12.0, 12.0, 12.0],
        'Mult_by_2_2.x_in.x': [4.0, 4.0, 4.0, 4.0, 4.666, 4.666, 4.666, 4.666, 5.332, 5.332, 5.332, 5.332, 6.0, 6.0, 6.0, 6.0]
    }
    assert (meta.state.doe_in.values == expected_doe_in.values).all()
    assert (meta.state.doe_in.columns == expected_doe_in.columns).all()
    assert meta.state.doe_out == expected_doe_out
    assert isinstance(meta.state.model, FloatKrigingSurrogate)
    assert isinstance(meta.state, SurrogateClassState)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_equation_warning(pme, cubic_DoE):
    dic = {'x_in.x': [0.5, 1., 1.5], 'u_in.x': [0.5, 1., 1.3333]}
    training_data = cubic_DoE(dic)
    pme.PMEMUG.make_surrogate(training_data, FloatKrigingSurrogate)
    pme.PMEMUG.x_in.x = 1.
    pme.PMEMUG.u_in.x = 1.
    with pytest.warns(LinAlgWarning, match="Singular matrix"):
        pme.run_drivers()


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_global_working_1(pme, cubic_DoE):
    dic = {'x_in.x': [0.5, 1., 1.5], 'u_in.x': [0.5, 1., 1.3333], 'SMXI.m.x' : numpy.linspace(23., 28., 6)}
    training_data = cubic_DoE(dic)
    nls_loc = pme.PMEMUG.S1E2M.Eq2u1.add_driver(NonLinearSolver('nls_loc'))
    rsc_loc = nls_loc.add_child(RunSingleCase('rsc_loc'))
    pme.PMEMUG.make_surrogate(training_data, FloatKrigingSurrogate)
    assert pme.PMEMUG._meta.state.doe_out is not None
    pme.PMEMUG.x_in.x = 1.
    pme.PMEMUG.u_in.x = 1.
    pme.run_drivers()
    assert pme.PMEMUG.S1E2M.Eq2u1.u == pytest.approx(0.1)
    assert pme.PMEMUG.SMXI.m.x == pytest.approx(26.75, rel=1e-4)
    assert pme.PMEMUG.x_out.x == pytest.approx(110., rel=1e-4)
    assert pme.PMEMUG.u_out.x == pytest.approx(111., rel=1e-4)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_global_working_3(sumg, training_data15):
    sumg.Eq_2Mult.make_surrogate(training_data15, FloatKrigingSurrogate)
    sumg.run_drivers()
    assert sumg.Eq_2Mult.Basic_Eq.x_in.x == pytest.approx(25.)
    assert sumg.Provider.x_in.x == pytest.approx(12.5, rel=1e-3)



def test_SurrogateClass_global_working_arrays(vectorsys, data_vectorsys_HF):
    vectorsys.make_surrogate(data_vectorsys_HF, FloatKrigingSurrogate)
    vectorsys["a"] = 1.5 * numpy.ones(3)
    vectorsys["in_.x"] = 0.75 * numpy.ones(3)
    vectorsys.run_drivers()
    assert vectorsys["out.x"] == pytest.approx(numpy.array([1.125, 1.125, 1.125]), rel = 2.e-1)    


def test_SurrogateClass_global_working_arrays2(vectorsplit, data_vectorsplit_HF):
    vectorsplit.make_surrogate(data_vectorsplit_HF, FloatKrigingSurrogate)
    vectorsplit["in_.x"] = numpy.ones(3)
    vectorsplit.run_drivers()
    assert vectorsplit["out1.x"] == pytest.approx(0.1 * numpy.ones(3), rel = 1.e-3)
    assert vectorsplit["out2.x"] == pytest.approx(0.9 * numpy.ones(3), rel = 1.e-3)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_dump_and_load(tmp_path, sumg, training_data15):
    sumg.Eq_2Mult.make_surrogate(training_data15, FloatKrigingSurrogate)
    meta = sumg.Eq_2Mult._meta
    meta.dump(tmp_path / "test_surrogate_save.obj")
    sumg.run_drivers()
    assert sumg.Provider.x_in.x == pytest.approx(12.5, rel=1e-3)

    #2ND SYSTEM
    SUMG2 = Sys_Unknown_1Eq_2Mult_Getter("SUMG2")
    meta2 = SurrogateClass(SUMG2.Eq_2Mult, None, FloatKrigingSurrogate)
    SUMG2.Eq_2Mult._meta = meta2
    meta2 = SurrogateClass.load(SUMG2.Eq_2Mult, tmp_path / "test_surrogate_save.obj")
    SUMG2.Eq_2Mult._meta = meta2
    nls2 = SUMG2.add_driver(NonLinearSolver('nls2'))
    rsc2 = nls2.add_child(RunSingleCase('rsc2'))
    #doe_in TEST
    assert (meta2.state.doe_in.columns == meta.state.doe_in.columns).all()
    assert (meta2.state.doe_in.values == meta.state.doe_in.values).all()
    #doe_out TEST
    assert meta2.state.doe_out == meta.state.doe_out
    #MODEL PREDICTION TEST
    for l in ([1.5, 0.5], [0.5, 10.], [5., 4.], [12.5, 11.5], [13., 7.]):
        assert (meta2.state.model.predict(numpy.array(l)) == meta.state.model.predict(numpy.array(l))).all()
    SUMG2.run_drivers()
    assert SUMG2.Provider.x_in.x == pytest.approx(12.5, rel=1e-3)


@pytest.mark.parametrize("wrong_state",[
    (15.),
    (dict()),
    (list()),
    (None),
    (numpy.ones(3)),
    ("sdqfs"),
])
def test_SurrogateClass_dump_and_load_error_state(tmp_path, sumg, wrong_state):
    sumg.Eq_2Mult.make_surrogate(None, FloatKrigingSurrogate)
    meta = sumg.Eq_2Mult._meta
    meta._SurrogateClass__state = wrong_state
    if isinstance(wrong_state, numpy.ndarray):
        assert (meta.state == wrong_state).all()
    else:
        assert meta.state == wrong_state
    meta.dump(tmp_path / "test_surrogate_save.obj")
    with pytest.raises(TypeError): 
        meta = SurrogateClass.load(meta.owner, tmp_path / "test_surrogate_save.obj")


@pytest.mark.parametrize("wrong_state",[
    (15.),
    (dict()),
    (list()),
    (None),
    (numpy.ones(3)),
    ("sdqfs"),
])
def test_SurrogateClass_dump_and_load_error_owner(tmp_path, sumg, wrong_state):
    meta = SurrogateClass(sumg.Eq_2Mult, None, FloatKrigingSurrogate)
    meta.dump(tmp_path / "test_surrogate_save.obj")
    with pytest.raises(TypeError): 
        meta = SurrogateClass.load(wrong_state, tmp_path / "test_surrogate_save.obj")


@pytest.mark.parametrize("system, expected_dict_path_str", list_for_get_dependant_connections)
def test_get_dependant_connections(system, expected_dict_path_str):
    assert get_dependant_connections(system) == expected_dict_path_str


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_SurrogateClass_no_output(no_output_sys, cubic_DoE):
    dic = {'a.a_in.x': [0.5, 1., 1.5], 'a.c.j.x': [0.5, 1., 1.3333]}
    training_data_ = cubic_DoE(dic)
    with pytest.raises(RuntimeError, match="no output found"):
        SurrogateClass(no_output_sys, training_data_, FloatKrigingSurrogate)


@pytest.mark.parametrize("data, pattern", [
    (
        {'a.a_in.x': [0.5, 1., 1.5], 'a.c.j.x': [0.5, 1., 1.3333]},
        ["b.inwards.u", "a.c.h"],
    ),
    (
        {'a.a_in.x': [0.5, 1., 1.5], 'a.c.j.x': [0.5, 1., 1.3333], 'b.inwards.u':[1., 2.]},
        ["a.c.h"],
    ),
    (
        {'a.a_in.x': [0.5, 1., 1.5], 'a.c.j.x': [0.5, 1., 1.3333], 'a.c.h':[1., 2.]},
        ["b.inwards.u"],
    ),
    (
        {'a.a_in.x': [0.5, 1., 1.5], 'a.c.j.x': [0.5, 1., 1.3333], 'a.c.h':[1., 2.], 'b.inwards.u':[1., 2.]},
        [],
    ),
])
def test_SurrogateClass_check_unknowns_and_transients(no_output_sys, DummyFactory, cubic_DoE, data, pattern):
    no_output_sys.add_child(
        DummyFactory('sub', outwards=get_args('u_out', 1.0, desc='blah')),
    )
    if data != {'a.a_in.x': [0.5, 1., 1.5], 'a.c.j.x': [0.5, 1., 1.3333], 'a.c.h':[1., 2.], 'b.inwards.u':[1., 2.]}:
        with pytest.warns(UserWarning) as records:
            SurrogateClass(no_output_sys, cubic_DoE(data), LinearNearestNeighbor)

        assert len(records) == 1
        message = records[0].message.args[0]
        for var in pattern:
            assert var in message
    else:
        SurrogateClass(no_output_sys, cubic_DoE(data), LinearNearestNeighbor)


def test_get_dependant_connections_with_NLS_inside(nude_sumg):
    nude_sumg.Eq_2Mult.Eq2u1.add_driver(NonLinearSolver("nls"))
    assert get_dependant_connections(nude_sumg) == {
        'Provider.u_out.x': [], 'Provider.x_out.x': [],
        'Eq_2Mult.x_in.x': [], 'Eq_2Mult.u_in.x': [],
        'Eq_2Mult.u_out.x': [], 'Eq_2Mult.x_out.x': [],
        'Eq_2Mult.Mult_by_2_1.x_in.x': [], 'Eq_2Mult.Mult_by_2_1.x_out.x': [],
        'Eq_2Mult.Mult_by_2_2.x_in.x': [], 'Eq_2Mult.Mult_by_2_2.x_out.x': [],
        'Eq_2Mult.Eq2u1.inwards.u': [], 'Eq_2Mult.Basic_Eq.x_in.x': [],
        'Get2D.x_in.x': [], 'Get2D.u_in.x': [],
        'Get2D.x_out.x': [], 'Get2D.u_out.x': []
    }


def test_SurrogateClass_check_unknowns_with_NLS_inside(nude_sumg, training_data3):
    nude_sumg.Eq_2Mult.Eq2u1.add_driver(NonLinearSolver("nls"))
    meta = nude_sumg.Eq_2Mult.make_surrogate(training_data3, FloatKrigingSurrogate)
    expected_keys = dict.fromkeys(['Basic_Eq.x_in.x', 'Eq2u1.inwards.u', 'Mult_by_2_1.x_in.x', 'Mult_by_2_1.x_out.x', 'Mult_by_2_2.x_in.x', 'Mult_by_2_2.x_out.x', 'u_out.x', 'x_out.x'])
    assert meta.state.doe_out.keys() == expected_keys.keys()


def test_SurrogateClass_check_unknowns_warning_without_NLS_inside(nude_sumg, training_data3):
    with pytest.warns(UserWarning):
        meta = nude_sumg.Eq_2Mult.make_surrogate(training_data3, FloatKrigingSurrogate)
        expected_keys = dict.fromkeys(['Basic_Eq.x_in.x', 'Mult_by_2_1.x_in.x', 'Mult_by_2_1.x_out.x', 'Mult_by_2_2.x_in.x', 'Mult_by_2_2.x_out.x', 'u_out.x', 'x_out.x'])
        assert meta.state.doe_out.keys() == expected_keys.keys()


def test_SurrogateClass_global_working2(surrogated_pme_LNN):
    data = [
        1.,   1.2,  1.5,  2.5,  0.5,
        -0.5, -1.5, -2.5, -2.3, -0.9,
    ]
    for d in data:
        surrogated_pme_LNN.PMEMUG.x_in.x = d
        surrogated_pme_LNN.run_drivers()
        assert surrogated_pme_LNN.PMEMUG.SMXI.m.x == pytest.approx(107./(4*d), rel = 5e-2)


################FOR SYSTEM##################
def test_System_set_recursive_active_status(p1e2mg):
    roG2D = p1e2mg.Get2D.add_driver(RunOnce('roG2D'))
    roP2D = p1e2mg.Provider.add_driver(RunOnce('roP2D'))
    nls = p1e2mg.Eq_2Mult.add_driver(NonLinearSolver('nls'))
    rsc = nls.add_child(RunSingleCase('rsc'))
    p1e2mg._set_recursive_active_status(True)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_make_surrogate_0(p1e2mg, data_in):
    meta = SurrogateClass(p1e2mg.Eq_2Mult, data_in, FloatKrigingSurrogate)
    p1e2mg.Eq_2Mult.make_surrogate(data_in, FloatKrigingSurrogate)
    assert p1e2mg.Eq_2Mult._meta is not None
    for vec in [numpy.array([1., 3.]), numpy.array([2., 5.]), numpy.array([0.5, 1.75]), numpy.array([1.8,3.79]), numpy.array([1000.,30.])]:
        assert (p1e2mg.Eq_2Mult._meta.state.model.predict(vec) == meta.state.model.predict(vec)).all()
    assert (p1e2mg.Eq_2Mult._meta.state.doe_in.columns == meta.state.doe_in.columns).all()
    assert (p1e2mg.Eq_2Mult._meta.state.doe_in.values == meta.state.doe_in.values).all()
    assert p1e2mg.Eq_2Mult._meta.state.doe_out == meta.state.doe_out
    assert p1e2mg.Eq_2Mult._meta_active_status


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_make_surrogate_execution_0(p1e2mg, data_in):
    with patch.multiple("cosapp.systems.surrogateclass.SurrogateClass",_SurrogateClass__prepare_and_train = DEFAULT) as mocked_methods:
        p1e2mg.Eq_2Mult.make_surrogate(data_in, FloatKrigingSurrogate)        
        mocked_methods['_SurrogateClass__prepare_and_train'].assert_called_once()


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_make_surrogate_global_working_transient(tmp_path, cubic_DoE):
    MySys = Sys_Lopped_Div_Int_Div('LoopSys')
    dic = {'h_in.x': numpy.linspace(-3, 10., 10), 'DI.v': numpy.linspace(-1., 4., 5), 'DI.h' : numpy.linspace(-1., 4., 5)}
    training_data_ = cubic_DoE(dic)
    MySys.DLDI.make_surrogate(training_data_, FloatKrigingSurrogate)
    MySys.DLDI.dump_surrogate(tmp_path / "myfile.obj")

    MySys2 = Sys_Lopped_Div_Int_Div('LoopSys2')
    MySys2.DLDI.load_surrogate(tmp_path / "myfile.obj")
    ee = MySys2.add_driver(EulerExplicit(dt=1, time_interval=[0, 10], order=2))
    nls = ee.add_child(NonLinearSolver('nls', factor=1.))
    rsc = nls.add_driver(RunSingleCase('rsc', verbose=False))
    ee.set_scenario(
        init = {'DLDI.DI.h': 'DLDI.DI.h0', 'DLDI.DI.v': 'DLDI.DI.v0'}
    )
    MySys2.run_drivers()

    MySys3 = Sys_Lopped_Div_Int_Div('LoopSys2')
    ee3 = MySys3.add_driver(EulerExplicit(dt=1, time_interval=[0, 10], order=2))
    nls3 = ee3.add_child(NonLinearSolver('nls', factor=1.))
    rsc3 = nls3.add_driver(RunSingleCase('rsc', verbose=False))
    ee3.set_scenario(
        init = {'DLDI.DI.h': 'DLDI.DI.h0', 'DLDI.DI.v': 'DLDI.DI.v0'}
    )
    MySys3.run_drivers()

    assert MySys2.DLDI.DI.h == pytest.approx(3.19, rel=1e-2)
    assert MySys3.DLDI.DI.h == pytest.approx(3.24, rel=1e-2)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_make_surrogate_1(pme, training_data_pme):
    meta = SurrogateClass(pme.PMEMUG, training_data_pme, FloatKrigingSurrogate)
    pme.PMEMUG.make_surrogate(training_data_pme, FloatKrigingSurrogate)
    assert pme.PMEMUG._meta is not None
    test_vectors = numpy.array([
        [1., 3., 8.],
        [2., 5., 47.],
        [0.5, 1.75, -10.],
        [1.8, 3.79, 5.],
        [1000., 30., -95.],
    ])
    for vec in test_vectors:
        assert pme.PMEMUG._meta.predict(vec) == pytest.approx(meta.predict(vec), rel=0.)
    assert (pme.PMEMUG._meta.state.doe_in.columns == meta.state.doe_in.columns).all()
    assert (pme.PMEMUG._meta.state.doe_in.values == meta.state.doe_in.values).all()
    assert pme.PMEMUG._meta.state.doe_out == meta.state.doe_out
    assert pme.PMEMUG._meta_active_status


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_make_surrogate_execution_1(pme, training_data_pme):
    with patch.multiple("cosapp.systems.surrogateclass.SurrogateClass", _SurrogateClass__prepare_and_train = DEFAULT) as mocked_methods:
        pme.PMEMUG.make_surrogate(training_data_pme, FloatKrigingSurrogate)        
        mocked_methods['_SurrogateClass__prepare_and_train'].assert_called_once()


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_make_surrogate_2(sumg, training_data3):
    meta = SurrogateClass(sumg.Eq_2Mult, training_data3, FloatKrigingSurrogate)
    sumg.Eq_2Mult.make_surrogate(training_data3, FloatKrigingSurrogate)
    assert sumg.Eq_2Mult._meta is not None
    for vec in [numpy.array([1., 3.]), numpy.array([2., 5.]), numpy.array([0.5, 1.75]), numpy.array([1.8,3.79]), numpy.array([1000.,30.])]:
        assert sumg.Eq_2Mult._meta.state.model.predict(vec) == pytest.approx(meta.state.model.predict(vec), rel=0.)
    assert (sumg.Eq_2Mult._meta.state.doe_in.columns == meta.state.doe_in.columns).all()
    assert (sumg.Eq_2Mult._meta.state.doe_in.values == meta.state.doe_in.values).all()
    assert sumg.Eq_2Mult._meta.state.doe_out == meta.state.doe_out
    assert sumg.Eq_2Mult._meta_active_status


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_make_surrogate_execution_2(sumg, training_data3):
    with patch.multiple("cosapp.systems.surrogateclass.SurrogateClass",_SurrogateClass__prepare_and_train = DEFAULT) as mocked_methods:
        sumg.Eq_2Mult.make_surrogate(training_data3, FloatKrigingSurrogate)        
        mocked_methods['_SurrogateClass__prepare_and_train'].assert_called_once()


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
@pytest.mark.parametrize("metatype, warn_err", [
    (True, UserWarning),
    (False, UserWarning),
    ("foo", UserWarning),
    (0, UserWarning),
    (1, UserWarning),
    (12., UserWarning),
    ({}, UserWarning),
    ([], UserWarning),
    (set(), UserWarning),
])
def test_System_make_surrogate_warning(sumg, training_data3, metatype, warn_err):
    sumg.Eq_2Mult._meta = metatype
    with pytest.warns(warn_err):
        sumg.Eq_2Mult.make_surrogate(training_data3, FloatKrigingSurrogate)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_dump_surrogate_load_surrogate_result(tmp_path, sumg, training_data15):
    sumg.Eq_2Mult.make_surrogate(training_data15, FloatKrigingSurrogate)
    sumg.Eq_2Mult.dump_surrogate(tmp_path / "sysmetadump.obj")
    sumg.Eq_2Mult._meta = None
    assert sumg.Eq_2Mult._meta is None
    sumg.Eq_2Mult.load_surrogate(tmp_path / "sysmetadump.obj")
    assert sumg.Eq_2Mult._meta_active_status
    assert sumg.Eq_2Mult._meta is not None
    sumg.run_drivers()
    assert sumg.Provider.x_in.x == pytest.approx(12.5, rel=1e-3)


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_dump_surrogate_load_surrogate_execution(tmp_path, sumg, training_data3):
    sumg.Eq_2Mult.make_surrogate(training_data3, FloatKrigingSurrogate)
    with patch('cosapp.systems.surrogateclass.SurrogateClass.dump') as mocked_dump:
        sumg.Eq_2Mult.dump_surrogate(tmp_path / "sysmetadump.obj")
        mocked_dump.assert_called_once()
    sumg.Eq_2Mult.dump_surrogate(tmp_path / "sysmetadump.obj")
    sumg.Eq_2Mult._meta = None
    assert sumg.Eq_2Mult._meta is None
    with patch('cosapp.systems.surrogateclass.SurrogateClass.load') as mocked_load:
        sumg.Eq_2Mult.load_surrogate(tmp_path / "sysmetadump.obj")
        mocked_load.assert_called_once()


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_dump_surrogate_load_surrogate_result1(tmp_path, pme, training_data_pme):
    MySystem = Sys_PME2MUG_G_1E('pme') #Declaring the 1st system
    MySystem.add_driver(NonLinearSolver('nls')).add_child(RunSingleCase('rsc')) #Adding some drivers
    MySystem2 = Sys_PME2MUG_G_1E('pme2') #Declaring the 2nd system
    MySystem2.add_driver(NonLinearSolver('nls2')).add_child(RunSingleCase('rsc2')) #Adding some drivers
    
    MySystem.PMEMUG.make_surrogate(training_data_pme, FloatKrigingSurrogate)
    MySystem.PMEMUG.dump_surrogate(tmp_path / "mysystem_pmemug.obj")

    MySystem2.PMEMUG.load_surrogate(tmp_path / "mysystem_pmemug.obj")
    #IMPORTANT : Let's make sure our 2 systems have the same inputs:

    MySystem2.PMEMUG.x_in.x = 1.0
    MySystem2.PMEMUG.u_in.x = 1.0
    MySystem2.PMEMUG.SMXI.m.x = 24.  #We start the resolution from the same point

    MySystem.PMEMUG.x_in.x = 1.0
    MySystem.PMEMUG.u_in.x = 1.0
    MySystem.PMEMUG.SMXI.m.x = 24.  #We start the resolution from the same point
    
    MySystem.run_drivers()
    MySystem2.run_drivers()

    assert MySystem.PMEMUG.SMXI.m.x == pytest.approx(26.75052, rel=1e-6)
    assert MySystem2.PMEMUG.SMXI.m.x == MySystem.PMEMUG.SMXI.m.x


def test_System_active_surrogate_RuntimeError(list_of_systems):
    for system_of_the_list in list_of_systems:
        with pytest.raises(RuntimeError):
            system_of_the_list.active_surrogate = True
            system_of_the_list.active_surrogate = False


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
def test_System_make_surrogate_DoE_out(pme, training_data_pme):
    Psync = {"x_out.x": list()}
    psync = list(Psync.keys())
    meta = pme.PMEMUG.make_surrogate(training_data_pme, FloatKrigingSurrogate, Psync = psync)
    assert meta.state.doe_out.keys() == Psync.keys()


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
@pytest.mark.parametrize("choice, error", [
    (True, None),
    (False, None),
    ("foo", TypeError),
    (0, TypeError),
    (1, TypeError),
    (12., TypeError),
    ({}, TypeError),
    ([], TypeError),
])
def test_System_active_surrogate_arg(sumg, training_data3, choice, error):
    sumg.Eq_2Mult.make_surrogate(training_data3, FloatKrigingSurrogate)
    if error is None:
        sumg.Eq_2Mult.active_surrogate = choice
        assert sumg.Eq_2Mult._meta_active_status == choice
    else:
        with pytest.raises(error):
            sumg.Eq_2Mult.active_surrogate = choice


@pytest.mark.filterwarnings("ignore:The.*unknowns/transients are not part of the training set")
@pytest.mark.parametrize("tag, expected", [
    (
        "Eq_2Mult",
        dict(
            meta_u = pytest.approx(14.8859, rel=1e-4), 
            meta_x = pytest.approx(13.3859, rel=1e-4),
            u = pytest.approx(14.5, rel=0.),
            x = pytest.approx(13.0, rel=0.),
        ),
    ),
    (
        "PMEMUG",
        dict(
            meta_u = pytest.approx(111.193, rel=1e-4),
            meta_x = pytest.approx(110, rel=1e-4),
            u = pytest.approx(111.333, rel=1e-4),
            x = pytest.approx(110, rel=1e-4),
        ),
    ),
    (
        "Provider",
        dict(
            meta_u = pytest.approx(15.0, rel=1e-5),
            meta_x = pytest.approx(12.5, rel=1e-5),
            u = pytest.approx(15.0, rel=0),
            x = pytest.approx(12.5),
        ),
    ),
])
def test_System_active_surrogate_result(case_factory, tag, expected):
    main, target = case_factory(tag)

    main.run_drivers()
    assert target.u_out.x == expected['meta_u']
    assert target.x_out.x == expected['meta_x']

    target.active_surrogate = False
    main.run_drivers()
    assert target.u_out.x == expected['u']
    assert target.x_out.x == expected['x']

    target.active_surrogate = True
    main.run_drivers()
    assert target.u_out.x == expected['meta_u']
    assert target.x_out.x == expected['meta_x']


################FOR DRIVER##################
def test_Driver_set_children_active_status_result(p1e2mg):
    roG2D = p1e2mg.Get2D.add_driver(RunOnce('roG2D'))
    roP2D = p1e2mg.Provider.add_driver(RunOnce('roP2D'))
    roE2M = p1e2mg.Eq_2Mult.add_driver(RunOnce('roE2M'))
    nls = p1e2mg.Eq_2Mult.add_driver(NonLinearSolver('nls'))
    rsc = nls.add_child(RunSingleCase('rsc'))
    roG2D._active = False
    roP2D._active = False
    nls._set_children_active_status(True)
    assert nls._active == True
    assert rsc._active == True
    #Verify that it doesn't touch not concerned drivers.
    assert roG2D._active == False
    assert roP2D._active == False
    #Some modifs. Verify that it doesn't touch not concerned drivers.
    nls._set_children_active_status(False)
    roG2D._set_children_active_status(True)
    assert nls._active == False
    assert rsc._active == False
    assert roP2D._active == False
    assert roG2D._active == True
    #One last modif. Verify that it doesn't touch not concerned drivers.
    roP2D._set_children_active_status(True)
    assert nls._active == False
    assert rsc._active == False
    assert roP2D._active == True
    assert roG2D._active == True
    assert roE2M._active == True