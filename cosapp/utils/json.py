import base64
import io
import json
import os
import pickle
import warnings
from typing import Any, Dict, Union, Optional, List

import jsonschema
import numpy
import pandas
from dataclasses import dataclass

from cosapp.utils.state_io import object__getstate__

JsonBaseType = Union[None, int, float, str, bool]
JsonType = Union[JsonBaseType, List[JsonBaseType], Dict[str, JsonBaseType]]


def jsonify(obj: Any) -> JsonType:
    """Converts an arbitrary object to a valid JSON type.

    Raise

    Parameters
    ----------
    obj : Any
        Object to convert

    Returns
    -------
    JsonType
        Conversion as a valid JSON type

    Raises
    ------
    TypeError
        If the object is not convertible
    """
    if isinstance(obj, bytes):
        return "data:text/plain;base64," + base64.b64encode(obj).decode("utf-8")
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return list(map(jsonify, obj))
    if isinstance(obj, (set, frozenset)):
        return list(map(jsonify, sorted(obj)))
    if isinstance(obj, dict):
        return {jsonify(key): jsonify(val) for key, val in obj.items()}
    if isinstance(obj, type):
        return {"__type__": f"{obj.__module__}.{obj.__qualname__}"}
    if hasattr(obj, "__json__"):
        j = jsonify(obj.__json__())
        j["__class__"] = f"{obj.__module__}.{obj.__class__.__qualname__}"
        return j
    if isinstance(obj, numpy.ndarray):
        buf = io.BytesIO()
        numpy.save(buf, obj)
        return "data:application/vnd.numpy.ndarray;base64," + base64.b64encode(
            buf.getvalue()
        ).decode("utf-8")
    if isinstance(obj, (numpy.int32, numpy.int64)):
        return obj
    if isinstance(obj, pandas.DataFrame):
        return obj.to_json()

    raise TypeError(f"Type {type(obj).__name__!r} cannot be JSONified")


class JSONEncoder(json.JSONEncoder):
    """Encode CoSApp object for JSON serialization."""

    def default(self, obj):
        try:
            return jsonify(obj)
        except TypeError:
            return super().default(obj)


def get_cosapp_type(class_name: str):
    import importlib
    from cosapp.base import System

    if class_name == "System":
        return System

    # check_arg(class_name, 'class_name', str, stack_shift=1)

    try:
        module_name, class_name = class_name.rsplit(".", maxsplit=1)
    except ValueError:
        module_name = ""

    if module_name:
        module = importlib.import_module(module_name)
    else:
        raise ImportError

    try:
        ty = getattr(module, class_name)
    except AttributeError:
        raise ImportError

    return ty


def decode_cosapp_dict(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON serialization back."""

    for key, value in document.items():
        if isinstance(value, dict):
            document[key] = decode_cosapp_dict(value)
        elif isinstance(value, str):
            if value.startswith("data:"):
                meta, data = value.split(",", maxsplit=1)
                meta = meta[5:].lower()

                if meta.endswith(";base64"):
                    data = base64.b64decode(bytes(data, encoding="utf-8"))

                if meta.startswith("text/plain;"):
                    document[key] = data
                elif meta.startswith("application/vnd.numpy.ndarray;"):
                    buf = io.BytesIO(data)
                    document[key] = numpy.load(buf)
                elif meta.startswith("application/vnd.python3.pickle;"):
                    document[key] = pickle.loads(data)
                else:
                    raise ValueError(f"Unknown data type {meta}.")

    return document


@dataclass
class EncodingMetadata:
    with_types: bool = True
    inputs_only: bool = False
    with_drivers: bool = True
    value_only: bool = False

    def __getstate__(self) -> Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
        """Creates a state of the object.

        The state type depend on the object, see
        https://docs.python.org/3/library/pickle.html#object.__getstate__
        for further details.

        Returns
        -------
        Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
            state
        """
        return object__getstate__(self)

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        return self.__getstate__().copy()

    def __iter__(self):
        return iter(self.__getstate__().values())


def _migrate_0_3_0(j):
    migrated = j.copy()

    assert len(migrated) == 1

    name, state = migrated.popitem()
    state["name"] = name
    state["__class__"] = state.pop("class")
    state["__encoding_metadata__"] = EncodingMetadata(
        with_types=True,
        inputs_only=True,
        with_drivers=False,
        value_only=True,
    ).__json__()

    inputs = {}
    for name, value in state.get("inputs", {}).items():
        port_name, var_name = name.split(".", maxsplit=1)

        if port_name not in inputs:
            inputs[port_name] = {"variables": {}}

        inputs[port_name]["variables"].update({var_name: value})

    if inputs:
        state["inputs"] = inputs

    for name, child in state.get("subsystems", {}).items():
        child = _migrate_0_3_0({name: child})

    return state


def from_json(j: dict[str, Any]):
    schema_id = j.pop("$schema", "")
    if schema_id:
        version, name = schema_id.split("/", maxsplit=1)
        path = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(path, "_".join((version, name)))) as fp:
            config_schema = json.load(fp)
        jsonschema.validate(j, config_schema)

        if version == "0-3-0":
            warnings.warn(
                f"Deprecated 'System' schema version {version}; please re-export the system using the 'to_json()' method",
                DeprecationWarning,
            )
            j = _migrate_0_3_0(j)

        if version < "0-3-0":
            raise ValueError(
                f"Schema {version} is not supported anymore (please use CoSApp<0.17.0)"
            )

    decoded_j = decode_cosapp_dict(j)

    if "__class__" in decoded_j:
        ty = get_cosapp_type(decoded_j["__class__"])
        obj = ty.load_from_dict(decoded_j)
        return obj

    return decoded_j


def loads_json(json_str: str):
    j = json.loads(json_str)
    return from_json(j)


def load_json(fp):
    j = json.load(fp)
    return from_json(j)


def to_json(obj) -> str:
    if hasattr(obj,"to_json"):
        return obj.to_json()
    if hasattr(obj,"__json__"):
        return json.dumps(obj.__json__())
    return json.dumps(obj)
