import numpy
from typing import Any, Dict, NoReturn, Iterable
import collections
import pickle
import pandas
from cosapp.utils.helpers import check_arg
from collections import OrderedDict
from typing import NamedTuple, Any
import logging
import warnings


logger = logging.getLogger(__name__)


class SurrogateClassState(NamedTuple):
    doe_in: pandas.DataFrame
    doe_out: OrderedDict
    model: Any
    doe_out_sizes: OrderedDict


class SurrogateClass:
    """
    This class is put around a System. It trains a metamodel thanks to input datas, based on the results of the system.
    """
    __slots__ = ('__owner', '__state')

    def __init__(self, owner, DoE_in, model: type, Psync = None, *args, **kwargs):
        logger.debug(f"Initialize surrogateclass for owner '{owner.name if owner is not None else None}', DoE_in as {DoE_in} and model as {model}")

        #CHECK ARGS
        check_arg(DoE_in, 'DoE_in', (pandas.DataFrame, type(None)))
        from cosapp.systems import System
        check_arg(owner, 'owner', (System, type(None)))
        if Psync is not None: check_arg(Psync, 'data to post synchronize', list)

        #BUILDING INTERN DATA
        self.__owner = owner
        doe_out = {s:list() for s in Psync} if Psync is not None else (OrderedDict() if owner is None else self.__get_owner_connections())
        model_obj = None if model is None else model(*args, **kwargs)
        self.__state = SurrogateClassState(DoE_in, doe_out, model_obj, OrderedDict())
        
        #TRAINING
        empty = any(obj is None for obj in (owner, DoE_in, model_obj))
        if not empty:
            self.__check_unknowns_and_transients()
            self.__prepare_and_train()

        logger.debug(f"surrogateclass initialized")


    @property
    def state(self)-> "SurrogateClassState":
        return self.__state


    @property
    def owner(self)-> "System":
        return self.__owner


    def __update_doe_out(self) -> NoReturn:
        logger.debug(f"Updating doe_out")
        owner = self.__owner
        doe_out = self.__state.doe_out
        for key in doe_out:
            doe_out[key].append(owner[key])
            logger.debug(f"added data to output {key} value is {doe_out[key]}")


    def __set_owner_inputs(self, dic_inputs) -> NoReturn:
        logger.debug(f"Setting {self.owner.name} input values")
        owner = self.__owner
        for key in dic_inputs:
            # logger.debug(f"set data to input {key} value is {dic_inputs[key]}")
            owner[key] = dic_inputs[key]


    def __get_doe_out_sizes(self):
        doe_out_sizes = OrderedDict.fromkeys(self.__state.doe_out.keys(), None)
        pos = 0
        length = 0
        for key in doe_out_sizes:
            if isinstance(self.__owner[key], numpy.ndarray):
                length = len(self.__owner[key].ravel())
            elif isinstance(self.__owner[key], (float, int)):
                length = 1
            doe_out_sizes[key] = (pos, length,)
            pos += length
        self.__state = SurrogateClassState(self.__state.doe_in, self.__state.doe_out, self.__state.model, doe_out_sizes)


    def __format_outputs(self) -> numpy.ndarray:
        logger.debug(f"Reshaping outputs")
        doe_length = len(self.__state.doe_in)
        doe_out = self.__state.doe_out
        reshaped_outputs = list()
        for k in range(doe_length):
            reshaped_outputs.append(
                list(value[k] for value in doe_out.values())
            )
        reshaped_outputs = numpy.asarray(list(list(flatten(el)) for el in reshaped_outputs))
        # logger.debug(f"Reshaped outputs are now numpy.ndarray and are : {reshaped_outputs}")
        return reshaped_outputs


    def __format_inputs(self) -> numpy.ndarray:
        logger.debug(f"Reshaping input datas")
        l = list()
        for couple_values in self.__state.doe_in.values:
            l.append(list(flatten(couple_values)))
        # logger.debug(f"Reshaped data is now numpy.ndarray and is : {numpy.asarray(l)}")
        return numpy.asarray(l)


    def __set_and_execute(self) -> NoReturn:
        logger.debug(f"Setting and executing in order to build datas for training")
        for label, row in self.state.doe_in.iterrows():
            self.__set_owner_inputs(row.to_dict(into=OrderedDict))#set ONE combination of inputs to the owner
            self.__owner.run_drivers()
            self.__update_doe_out()#adds output datas to 'meta_dico_out'


    def add_data(self, newdoe : pandas.DataFrame) -> NoReturn:
        #it should merge input lists when matching names #TODO this method doesn't work yet.
        pass
        self.__state = SurrogateClassState(self.__state.doe_in, self.__state.doe_out, self.__state.model, OrderedDict())


    def __prepare_and_train(self) -> NoReturn:
        logger.debug(f"Preparing and training function")
        if len(self.state.doe_out) == 0:
            raise RuntimeError(f"Cannot train surrogate model: no output found in System {self.owner.name!r}")
        self.__get_doe_out_sizes()
        self.__set_and_execute()
        x = self.__format_inputs()
        y = self.__format_outputs()
        self.__train_model(x, y)


    def __train_model(self, x, y) -> NoReturn:
        logger.debug(f"Training model {self.state.model} with X as :\n{x} \nand Y as {y}")
        self.__state.model.train(x,y)
        logger.debug(f"Model {self.state.model} trained")


    def predict(self, x):
        logger.debug(f"Predicting value of {x} with model {self.state.model}")
        return self.__state.model.predict(x).reshape(1,-1)[0]


    def compute(self) -> NoReturn:
        logger.debug(f"Starting meta_compute() instead of {self.owner.name}'s compute().")
        current_inputs = self.__get_current_inputs()
        current_outputs = self.predict(current_inputs)
        self.__set_current_outputs(current_outputs)
        logger.debug(f"Ending meta_compute()")


    def __get_current_inputs(self) -> numpy.ndarray:
        logger.debug(f"Getting current inputs of system {self.owner.name}")
        L = list(flatten(self.__owner[key] for key in self.__state.doe_in.columns))
        logger.debug(f"Detected inputs valus are : {L}")
        return numpy.asarray(L)


    def __set_current_outputs(self, np_outputs : numpy.ndarray) -> NoReturn:
        logger.debug(f"Setting current outputs of system {self.owner.name}")
        for key, value in self.__state.doe_out_sizes.items():
            pos = value[0]
            length = value[1]
            if length == 1:
                self.__owner[key] = np_outputs[pos]
            elif length > 1:
                self.__owner[key].ravel()[:] = np_outputs[pos:pos+length]
        

    def dump(self, filename: str) -> NoReturn:
        logger.debug(f"Dumping metamodel in {filename}")
        with open(filename, 'wb') as fp:
            pickle.dump(self.__state, fp)


    @classmethod
    def load(cls, owner, filename: str) -> "SurrogateClass":
        logger.debug(f"Loading metamodel from {filename}")
        from cosapp.systems import System
        check_arg(owner, 'owner', System)
        with open(filename, 'rb') as fp:
            state = pickle.load(fp)
        if not isinstance(state, SurrogateClassState):
            raise TypeError("cannot load object")
        obj = cls(None, None, None)
        obj.__owner = owner
        obj.__state = state
        return obj


    def __check_unknowns_and_transients(self) -> NoReturn:
        names = get_unknows_transients(self.__owner)
        unsolvable_unknows = set(names).difference(self.state.doe_out.keys())
        trained_unknowns = unsolvable_unknows.issubset(self.state.doe_in.keys())
        if not trained_unknowns:
            message = "The following unknowns/transients are not part of the training set; "\
                "future attempts to compute them with a driver may fail:"
            for name in names.difference(self.state.doe_in.keys()):
                message += f"\n\t- {name}"
            warnings.warn(message)


    def __get_owner_connections(self) -> Dict[str, list]:
        return get_dependant_connections(self.__owner)



def flatten(iterable) -> Iterable:
    for el in iterable:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, str): 
            yield from flatten(el)
        else:
            yield el


def get_dependant_connections(system, head_system = None) -> Dict[str, list]:
    """
    This function returns a dictionnary.
    Keys are paths to connected inputs and all outputs.
    Values are empty lists.
    """
    logger.debug(f"Starting recursive getter of connected inputs and all outputs on {system.name}")
    dict_path_str = OrderedDict()
    prefix = ""
    if head_system is None:
        head_system = system
    elif system is not head_system:
        prefix = f"{head_system.get_path_to_child(system)}."
        for list_connectors in system.parent._connectors.values():
            for connector in list_connectors:
                var_sink = connector.sink
                logger.debug(f"Detecting connector {connector} with sink {var_sink}")
                if var_sink in system.inputs.values():
                    for portvariable in var_sink._variables :
                        dict_path_str[f"{prefix}{connector.sink.name}.{portvariable}"] = list()
                        logger.debug(f"Adding {prefix}{connector.sink.name}.{portvariable} to list of connected inputs")

    for output in system.outputs.values():
        logger.debug(f"Checking output {output}.")
        for portvariable in output._variables:
            logger.debug(f"Checking variable {portvariable} of output {output}.")
            dict_path_str[f"{prefix}{output.name}.{portvariable}"] = list()
    if system.is_standalone():
        unknowns = system.get_unsolved_problem().unknowns
        for unknown in unknowns:
            dict_path_str[f"{prefix}{unknown}"] = list()
    for child in system.children.values():
        logger.debug(f"Targeted child of recursive getter of connected inputs and all outputs is {child}")
        child_dict_path_str = get_dependant_connections(child, head_system)
        dict_path_str.update(child_dict_path_str)
    return dict_path_str


def get_unknows_transients(system) -> set:
    res = set()
    problem = system.get_unsolved_problem()
    for collection in (problem.unknowns, problem.transients):
        res |= set(var for var in collection)
    
    return res
