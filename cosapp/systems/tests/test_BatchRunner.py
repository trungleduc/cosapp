import pytest
import numpy as np
import pandas as pd
import itertools

from cosapp.systems.batch import BatchRunner
from cosapp.core.execution import ExecutionPolicy, ExecutionType, get_start_methods
from cosapp.tests.library.systems import Resistor, Node, Ground
from cosapp.systems import System
from cosapp.drivers import NonLinearSolver


class DummySystem(System):
    def setup(self):
        self.add_inward("a", 0.0)
        self.add_inward("x", 0.0)
        self.add_outward("y", 0.0)
        self.add_outward("z", 0.0)

    def compute(self):
        self.y = self.x * 2
        self.z = self.x ** 2 - self.a


class Circuit(System):
    """Simple resistor circuit

    --->--+--[R2]--+
          |        |
        [R1]     [R3]
          |        |
          +--------+---/ Ground
    """
    def setup(self):
        ground = self.add_child(Ground("ground", V=0.0))
        R1 = self.add_child(Resistor("R1", R=1000))
        R2 = self.add_child(Resistor("R2", R=500))
        R3 = self.add_child(Resistor("R3", R=250))
        
        Node.make(self, "n1",
            incoming = [],
            outgoing = [R1, R2],
            pulling = {"I_in0": "source"},
        )
        Node.make(self, "n2",
            incoming = [R2],
            outgoing = [R3],
        )
        self.connect(ground.V_out, R1.V_out)
        self.connect(ground.V_out, R3.V_out)

        self.add_outward("Req", 0.0, desc="Equivalent resistance of the circuit")

    def compute(self) -> float:
        self.Req = self.n1.V / self.source.I

    def Req_exact(self) -> float:
        """Equivalent resistance of the circuit (exact solution)"""
        R1, R2, R3 = (self[f"R{n}.R"] for n in range(1, 4))
        return 1 / (1 / R1 + 1 / (R2 + R3))


@pytest.fixture
def dummy():
    return DummySystem("dummy")


@pytest.fixture
def circuit():
    circuit = Circuit("circuit")
    circuit.R1.R = 1e3
    circuit.R2.R = 0.5e3
    circuit.R3.R = 0.25e3
    circuit.source.I = 0.1
    return circuit


@pytest.fixture
def output_varnames():
    return ["y"]


@pytest.fixture
def batch_runner(dummy, output_varnames):
    return BatchRunner(dummy, output_varnames)


def test_BatchRunner_initialization(dummy):
    runner = BatchRunner(dummy, ["y"])
    assert isinstance(runner, BatchRunner)
    assert runner.system is dummy
    assert runner.output_varnames == ("y",)
    assert isinstance(runner.execution_policy, ExecutionPolicy)


def test_BatchRunner_output_varnames(batch_runner):
    assert batch_runner.output_varnames == ("y",)
    batch_runner.output_varnames = ["y", "z"]
    assert batch_runner.output_varnames == ("y", "z")


@pytest.mark.parametrize("includes, excludes, expected", [
    ([], [], ()),
    ("y", [], ("y",)),
    (["y"], [], ("y",)),
    (["y", "z"], [], ("y", "z")),
    (["y", "z"], ["z"], ("y",)),
    (["*"], [], ("y", "z")),
    (["?"], [], ("y", "z")),
    (["*"], ["x"], ("y", "z")),
    (["*"], ["a"], ("y", "z")),
])
def test_BatchRunner_find_outputs(dummy, includes, excludes, expected):
    """Test methods `find_outputs` and `from_output_pattern`."""
    runner = BatchRunner(dummy, [])
    assert runner.output_varnames == ()

    runner.find_outputs(includes, excludes)
    assert runner.output_varnames == expected

    runner = BatchRunner.from_output_pattern(dummy, includes, excludes)
    assert isinstance(runner, BatchRunner)
    assert runner.output_varnames == expected


def test_BatchRunner_execution_policy(batch_runner):
    policy = ExecutionPolicy(2, ExecutionType.MULTI_PROCESSING)
    batch_runner.execution_policy = policy
    assert batch_runner.execution_policy is policy

    with pytest.raises(TypeError):
        batch_runner.execution_policy = "invalid_policy"


@pytest.mark.parametrize(
    "varnames, expected", [
        (["a", "b", "c"], {"a": [], "b": [], "c": []}),
        (["x", "y", "z"], {"x": [], "y": [], "z": []}),
        (["x", "y", "foo.z"], {"x": [], "y": [], "foo.z": []}),
        (["a", "b", "a"], {"a": [], "b": []}),
        (["a"], {"a": []}),
        ([], {}),
    ]
)
def test_BatchRunner_empty_dataset(varnames, expected):
    assert BatchRunner.empty_dataset(varnames) == expected


@pytest.mark.parametrize("n_procs", [1, 2])    
@pytest.mark.parametrize("start_method", get_start_methods())
@pytest.mark.parametrize(
    "inputs, output_varnames, expected", [
        (
            pd.DataFrame({"x": [0.5, 2.0, 0.5, 0.1], "a": [-1.0, 0.4, -0.1, -0.1]}),
            ["y", "z"],
            {"y": [1.0, 4.0, 1.0, 0.2], "z": [1.25, 3.6, 0.35, 0.11]},
        ),
        (
            pd.DataFrame({"x": [0.5, 2.0, 0.5, 0.1], "a": [-1.0, 0.4, -0.1, -0.1]}),
            ["z"],
            {"z": [1.25, 3.6, 0.35, 0.11]},
        ),
    ]
)
def test_BatchRunner_run(dummy, inputs, n_procs, start_method, output_varnames, expected):
    execution_policy = ExecutionPolicy(n_procs, ExecutionType.MULTI_PROCESSING, start_method)
    runner = BatchRunner(dummy, output_varnames, policy=execution_policy)
    
    results = runner.run(inputs)

    assert set(results.keys()) == set(output_varnames)
    assert set(results.keys()) == set(runner.output_varnames)

    for varname in results:
        assert results[varname] == pytest.approx(expected[varname], rel=1e-12), f"Mismatch for {varname=!r}"


@pytest.mark.parametrize("n_procs", [2, 3])    
@pytest.mark.parametrize("start_method", get_start_methods())
def test_BatchRunner_run_with_driver(circuit, n_procs, start_method):

    circuit.source.I = 1.0
    circuit.R2.R = 0.5e3

    solver = circuit.add_driver(NonLinearSolver("solver"))

    circuit.run_drivers()  # equilibrate the circuit. first

    # Set design problem
    k = 0.6
    solver.add_unknown("R2.R").add_equation(f"Req == {k} * R1.R")

    axes = {
        "R1.R": np.linspace(500, 2500, 11),
        "R3.R": np.linspace(100, 400, 4),
    }
    inputs = pd.DataFrame(
        itertools.product(*axes.values()),
        columns=axes.keys(),
    )

    runner = BatchRunner(circuit, output_varnames=["R2.R"])
    execution_policy = ExecutionPolicy(n_procs, ExecutionType.MULTI_PROCESSING, start_method)
    
    outputs = runner.run(inputs, policy=execution_policy)

    assert set(outputs.keys()) == {"R2.R"}

    R1 = np.asarray(inputs["R1.R"])
    R3 = np.asarray(inputs["R3.R"])
    R2 = np.asarray(outputs["R2.R"])

    assert len(R2) == len(inputs)
    assert R2 == pytest.approx(k / (1 - k) * R1 - R3)
