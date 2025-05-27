from __future__ import annotations
import numpy
import pickle
import pandas
import logging
import warnings
from numbers import Number
from collections import OrderedDict
from collections.abc import Iterable, Generator
from typing import (
    Any, Union,
    Optional, NamedTuple,
    TYPE_CHECKING,
)
if TYPE_CHECKING:
    from cosapp.systems import System

from cosapp.systems.batch import BatchRunner
from cosapp.utils.surrogate_models.base import SurrogateModel
from cosapp.ports.enum import PortType
from cosapp.utils.logging import LogLevel
from cosapp.utils.helpers import check_arg
from cosapp.utils.find_variables import find_variables, make_wishlist, natural_varname
from cosapp.core.execution import ExecutionPolicy, ExecutionType


logger = logging.getLogger(__name__)


class SurrogateModelProxy(SurrogateModel):
    """Surrogate model proxy used in `SystemSurrogate`.
    Adds an internal property `trained`, and check that model
    has been trained before each prediction.
    """
    def __init__(self, model):
        if not isinstance(model, SurrogateModel):
            raise TypeError(
                "`SurrogateModelProxy` can only wrap `SurrogateModel` instances."
            )
        self.__wrappee = model
        self.__trained = False

    @property
    def trained(self) -> bool:
        return self.__trained

    def train(self, x, y) -> None:
        """Trains surrogate model with inputs `x` and outputs `y`.

        Parameters
        ----------
        x : array-like
            Training input locations
        y : array-like
            Model responses at given inputs.
        """
        model = self.__wrappee
        logger.debug(f"Training model {model} with \n\tX = {x}\n\tY = {y}")
        model.train(x, y)
        self.__trained = True
        logger.debug(f"Model {model} trained")

    def predict(self, x) -> numpy.ndarray:
        """Calculates a predicted value of the response based on the current trained model.

        Parameters
        ----------
        x : array-like
            Point(s) at which the surrogate is evaluated.
        """
        model = self.__wrappee
        if not self.__trained:
            raise RuntimeError(
                f"{type(model).__name__} has not been trained, so no prediction can be made."
            )
        return model.predict(x)

    def get_type(self) -> type[SurrogateModel]:
        """Returns wrapped model type"""
        return type(self.__wrappee)


class SystemSurrogateState(NamedTuple):
    """Named tuple containing internal state data of a SystemSurrogate object"""
    doe_in: pandas.DataFrame
    doe_out: Union[pandas.DataFrame, OrderedDict]
    model: SurrogateModelProxy
    doe_out_sizes: OrderedDict

    def input_varnames(self) -> Iterable[str]:
        return self.doe_in.keys()

    def output_varnames(self) -> Iterable[str]:
        return self.doe_out.keys()


class SystemSurrogate:
    """
    Class representing a surrogate model trained from the outputs of an owner system.
    Once the surrogate is created, it supersedes (whenever activated) the behaviour of
    `System.compute()`. 
    """
    __slots__ = ("__owner", "__state", "__need_doe", "__exec_policy")

    def __init__(self,
        owner: System,
        data_in: Union[pandas.DataFrame, dict[str, list]],
        model: type[SurrogateModel],
        data_out: Optional[Union[pandas.DataFrame, dict[str, list]]] = None,
        postsynch: Union[str, list[str]] = "*",
        execution_policy: Optional[ExecutionPolicy] = None,
        *args, **kwargs,
    ):
        # CHECK ARGS
        from cosapp.systems import System
        check_arg(owner, "owner", System)
        check_arg(data_in, "data_in", (pandas.DataFrame, dict))
        check_arg(data_out, "data_out", (pandas.DataFrame, dict, type(None)))
        if model is not None:
            try:
                check_arg(model, "model", type, lambda m: issubclass(m, SurrogateModel))
            except ValueError:
                raise ValueError(
                    f"`model` must be a concrete implementation of `SurrogateModel`"
                    f"; got `{model.__name__}`",
                )
            else:
                model_obj = SurrogateModelProxy(model(*args, **kwargs))
        else:
            model_obj = None

        postsynch = make_wishlist(postsynch, "postsynch")
        
        # BUILD INTERN DATA
        self.__owner: System = owner
        self.__need_doe = (data_out is None)
        self.__exec_policy = ExecutionPolicy(1, ExecutionType.SINGLE_THREAD)
        if execution_policy:
            self.execution_policy = execution_policy
        
        doe_in = pandas.DataFrame.from_dict(data_in) if isinstance(data_in, dict) else data_in
        doe_out = self.__init_doe_out(data_out, postsynch)
        self.filter_headers(doe_in)
        self.__state = SystemSurrogateState(doe_in, doe_out, model_obj, OrderedDict())

        # TRAINING
        empty = owner is None or model_obj is None or len(data_in) == 0
        if not empty:
            logger.debug(
                f"Initialize {model.__name__} surrogate for System {owner.name!r}"
                f", with {len(doe_in)} samples of {list(doe_in.columns)}"
            )
            self.__check_unknowns_and_transients()
            self.__prepare_and_train()

        logger.debug(f"System surrogate initialized")

    @property
    def owner(self) -> System:
        return self.__owner

    @property
    def state(self) -> SystemSurrogateState:
        """SystemSurrogateState: inner state of system surrogate"""
        return self.__state

    @property
    def model_type(self) -> type[SurrogateModel]:
        return self.__state.model.get_type()

    @property
    def trained(self) -> bool:
        return self.__state.model.trained

    @property
    def synched_outputs(self) -> list[str]:
        """list[str]: list of synchronized output variable names"""
        return list(self.__state.doe_out_sizes)

    @staticmethod
    def filter_headers(doe: pandas.DataFrame) -> None:
        """Apply `natural_varname` to dataframe column names"""
        mapping = dict(
            (name, natural_varname(name))
            for name in doe.columns
        )
        doe.rename(mapping, axis=1, inplace=True)

    @property
    def execution_policy(self) -> ExecutionPolicy:
        return self.__exec_policy

    @execution_policy.setter
    def execution_policy(self, execution_policy: ExecutionPolicy):
        check_arg(execution_policy, "execution_policy", ExecutionPolicy)
        self.__exec_policy = execution_policy

    def __init_doe_out(self, data_out, postsynch) -> dict[str, list]:
        doe_out = OrderedDict()

        if isinstance(data_out, pandas.DataFrame):
            doe_out = data_out.to_dict(into=OrderedDict)

        elif data_out is not None:
            doe_out = OrderedDict(data_out)

        elif self.owner is not None:
            if isinstance(postsynch, str):
                postsynch = [postsynch]
            if "*" in postsynch:
                watched = self.__get_owner_connections()
                varnames = watched.keys()
            else:
                owner = self.owner
                def writeable(varname: str) -> bool:
                    try:
                        owner[varname] = owner[varname]
                    except AttributeError:
                        return False
                    else:
                        return True
                matches = find_variables(
                    owner,
                    includes=postsynch,
                    excludes=None,
                    inputs=False,
                )
                varnames = set(filter(writeable, matches.keys()))
                # Make sure varlist contains at least owner system outputs
                for portname, port in owner.outputs.items():
                    varnames.update(natural_varname(f"{portname}.{varname}") for varname in port)
            doe_out = OrderedDict((name, []) for name in varnames)

        return doe_out

    def __get_doe_out_sizes(self) -> None:
        state = self.__state
        owner = self.__owner
        doe_out_sizes = OrderedDict.fromkeys(state.output_varnames(), None)
        pos = 0
        for varname in doe_out_sizes:
            size = 0
            attr = getattr(owner, varname)
            if isinstance(attr, (Number, numpy.ndarray)):
                size = numpy.size(attr)
            else:
                raise TypeError(
                    f"Unsupported data type {type(attr).__name__} for {owner.name}.{varname}"
                )
            doe_out_sizes[varname] = (pos, size)
            pos += size

        self.__state = state._replace(doe_out_sizes=doe_out_sizes)

    def __format_outputs(self) -> numpy.ndarray:
        logger.debug(f"Reshaping outputs")
        state = self.__state
        reshaped_outputs = []
        for k in range(len(state.doe_in)):
            reshaped_outputs.append(
                list(value[k] for value in state.doe_out.values())
            )
        reshaped_outputs = numpy.asarray(list(list(flatten(el)) for el in reshaped_outputs))
        # logger.debug(f"Reshaped outputs are now numpy.ndarray and are: {reshaped_outputs}")
        return reshaped_outputs

    def __format_inputs(self) -> numpy.ndarray:
        logger.debug(f"Reshaping input data")
        res = []
        for couple_values in self.__state.doe_in.values:
            res.append(list(flatten(couple_values)))
        return numpy.asarray(res)

    def __set_and_execute(self) -> None:
        logger.debug(f"Setting and executing in order to build data for training")
        owner = self.__owner
        state = self.__state
        output_varnames = tuple(state.output_varnames())

        runner = BatchRunner(owner, output_varnames, policy=self.execution_policy)
        outputs = runner.run(state.doe_in)
        self.__state = state._replace(doe_out=outputs)

    def add_data(self, new_doe: pandas.DataFrame) -> None:
        # TODO: Should merge input lists when matching names (not working yet)
        # self.__state = SystemSurrogateState(
        #     self.__state.doe_in,
        #     self.__state.doe_out,
        #     self.__state.model,
        #     OrderedDict(),
        # )
        raise NotImplementedError()

    def __prepare_and_train(self) -> None:
        logger.debug(f"Preparing and training function")
        if len(self.__state.doe_out) == 0:
            raise ValueError(
                f"Cannot train surrogate model: no output found in System {self.owner.name!r}"
            )
        self.__get_doe_out_sizes()
        if self.__need_doe:
            self.__set_and_execute()
        x = self.__format_inputs()
        y = self.__format_outputs()
        self.__train_model(x, y)

    def __train_model(self, x, y) -> None:
        self.__state.model.train(x, y)

    def predict(self, x):
        return self.__state.model.predict(x).ravel()

    def compute(self) -> None:
        logger.debug(f"Start meta_compute() instead of {self.owner.name}.compute().")
        inputs = self.__get_owner_inputs()
        outputs = self.predict(inputs)
        self.__set_owner_outputs(outputs)
        logger.debug(f"meta_compute() done")

    def __get_owner_inputs(self) -> numpy.ndarray:
        owner = self.owner
        logger.debug(f"Getting current inputs of system {owner.name}")
        inputs = list(flatten(owner[varname] for varname in self.__state.doe_in.columns))
        logger.debug(f"Detected input values: {inputs}")
        return numpy.asarray(inputs)

    def __set_owner_outputs(self, np_outputs: numpy.ndarray) -> None:
        owner = self.__owner
        logger.log(LogLevel.FULL_DEBUG, f"Post-synchronize outputs of {owner.name}")
        for varname, (pos, size) in self.__state.doe_out_sizes.items():
            var = owner.get_variable(varname)
            attr = var.value
            if isinstance(attr, numpy.ndarray):
                attr.flat = np_outputs[pos : pos + size]
            elif size == 1:
                owner[varname] = np_outputs[pos]
            else:
                # Should never occur, according to `__get_doe_out_sizes`
                raise TypeError(
                    f"Unsupported data type {type(attr).__name__} for {owner.name}.{varname}"
                )

    def dump(self, filename: str) -> None:
        """Dump current state to file"""
        logger.debug(f"Dumping metamodel in {filename}")
        with open(filename, "wb") as fp:
            pickle.dump(self.__state, fp)

    @classmethod
    def load(cls, owner: System, filename: str) -> SystemSurrogate:
        """Load system surrogate from file"""
        logger.debug(f"Loading metamodel from {filename}")
        from cosapp.systems import System
        check_arg(owner, "owner", System)
        with open(filename, "rb") as fp:
            state = pickle.load(fp)
        if not isinstance(state, SystemSurrogateState):
            raise TypeError("cannot load object")
        obj = cls(owner, data_in={}, model=None)
        obj.__state = state
        return obj

    def __check_unknowns_and_transients(self) -> None:
        state = self.__state
        unknowns = get_unknowns_transients(self.__owner)
        unsolvable_unknowns = set(unknowns).difference(state.doe_out)
        trained_unknowns = unsolvable_unknowns.issubset(state.doe_in)
        if not trained_unknowns:
            warnings.warn(
                "The following unknowns/transients are not part of the training set; "
                f"future attempts to compute them with a driver may fail: "
                f"{list(unknowns.difference(state.doe_in))}"
            )

    def __get_owner_connections(self) -> dict[str, list]:
        return get_dependent_connections(self.__owner)


def flatten(iterable: Iterable) -> Generator[Any, None, None]:
    for elem in iterable:
        if isinstance(elem, Iterable) and not isinstance(elem, str): 
            yield from flatten(elem)
        else:
            yield elem


def get_dependent_connections(system: System) -> dict[str, PortType]:
    """This function returns a dictionnary mapping variable names to a port direction.
    Keys are absolute paths to connected inputs and all outputs.
    Values are owner port direction.
    """
    def get_connections(system: System, head_system: System) -> dict[str, PortType]:
        """Recursive inner version of `get_dependent_connections`"""
        result = dict()
        prefix = ""
        if system is not head_system:
            prefix = f"{head_system.get_path_to_child(system)}."
            for connector in system.incoming_connectors():
                sink = connector.sink
                logger.debug(f"Detecting connector {connector} with sink {sink.name!r}")
                if sink.is_input:
                    for var in connector.sink_variables():
                        key = f"{prefix}{sink.name}.{var}"
                        result[key] = PortType.IN
                        logger.debug(f"Add {key} to list of connected inputs")

        for output in system.outputs.values():
            logger.debug(f"Checking output {output}.")
            for var in output:
                key = f"{prefix}{output.name}.{var}"
                result[key] = PortType.OUT
                logger.debug(f"Add {key} to list of outputs")
        if system.is_standalone():
            unknowns = system.assembled_problem().unknowns
            for unknown in unknowns:
                key = f"{prefix}{unknown}"
                result[key] = PortType.IN
                logger.debug(f"Add {key} to list of unknowns")
        for child in system.children.values():
            logger.debug(
                f"Targetted child of recursive getter of connected inputs and all outputs is {child}"
            )
            result.update(get_connections(child, head_system))
        return result

    logger.debug(
        f"Recursive search of connected inputs and all outputs on {system.name}"
    )
    return get_connections(system, system)


def get_unknowns_transients(system: System) -> set[str]:
    return set.union(
        set(system.assembled_problem().unknowns),
        set(system.assembled_time_problem().transients),
    )
