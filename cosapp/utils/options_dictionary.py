"""Define the OptionsDictionary class.

Copyright (c) 2016-2018, openmdao.org

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module comes from OpenMDAO 2.2.0. It was slightly modified for CoSApp integration.
"""

from __future__ import annotations

# unique object to check if default is given

from typing import Any, Dict, Union, Optional, Callable, Container, Iterable
from cosapp.utils.state_io import object__getstate__


_undefined = object()


class OptionsDictionary(object):
    """
    Dictionary with pre-declaration of keys for value-checking and default values.

    This class is instantiated for:
        1. the options attribute in solvers, drivers, and processor allocators
        2. the supports attribute in drivers
        3. the options attribute in systems

    Attributes
    ----------
    _dict : dict of dict
        Dictionary of entries. Each entry is a dictionary consisting of value, values,
        dtype, desc, lower, and upper.
    _read_only : bool
        If True, no options can be set after declaration.
    """

    __slots__ = ["_dict", "_read_only"]

    def __init__(self, read_only=False):
        """
        Initialize all attributes.

        Parameters
        ----------
        read_only : bool
            If True, setting (via __setitem__ or update) is not permitted.
        """
        self._dict = {}
        self._read_only = read_only

    def __getstate__(
        self,
    ) -> Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
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
        qualname = f"{self.__module__}.{self.__class__.__qualname__}"
        d, state = self.__getstate__()

        return {"__class__": qualname, "state": state}

    def __repr__(self):
        """
        Return a dictionary representation of the options.

        Returns
        -------
        dict
            The options dictionary.
        """
        return repr(self._dict)

    def __rst__(self):
        """
        Generate reStructuredText view of the options table.

        Returns
        -------
        list of str
            A rendition of the options as an rST table.
        """
        outputs = []
        for option_name, option_data in sorted(self._dict.items()):
            name = option_name
            default = (
                option_data["value"]
                if option_data["value"] is not _undefined
                else "**Required**"
            )
            values = option_data["values"]
            dtype = option_data["dtype"]
            desc = option_data["desc"]

            # if the default is an object instance, replace with the (unqualified) object type
            default_str = str(default)
            idx = default_str.find(" object at ")
            if idx >= 0 and default_str.startswith("<"):
                parts = default_str[:idx].split(".")
                default = parts[-1]

            if dtype is None:
                dtype = "N/A"
            elif dtype is not None:
                if not isinstance(dtype, (tuple, list)):
                    dtype = (dtype,)
                dtype = [type_.__name__ for type_ in dtype]

            if values is None:
                values = "N/A"

            elif values is not None:
                if not isinstance(values, (tuple, list)):
                    values = (values,)

                values = [value for value in values]

            outputs.append([name, default, values, dtype, desc])

        lines = []

        col_heads = [
            "Option",
            "Default",
            "Acceptable Values",
            "Acceptable Types",
            "Description",
        ]

        max_sizes = {}
        for j, col in enumerate(col_heads):
            max_sizes[j] = len(col)

        for output in outputs:
            for j, item in enumerate(output):
                length = len(str(item))
                if max_sizes[j] < length:
                    max_sizes[j] = length

        header = ""
        titles = ""
        for key, val in max_sizes.items():
            header += "=" * val + " "

        for j, head in enumerate(col_heads):
            titles += "%s " % head
            size = max_sizes[j]
            space = size - len(head)
            if space > 0:
                titles += space * " "

        lines.append(header)
        lines.append(titles)
        lines.append(header)

        n = 3
        for output in outputs:
            line = ""
            for j, item in enumerate(output):
                line += "%s " % str(item)
                size = max_sizes[j]
                space = size - len(str(item))
                if space > 0:
                    line += space * " "

            lines.append(line)
            n += 1

        lines.append(header)

        return lines

    def __str__(self, width=100):
        """
        Generate text string representation of the options table.

        Parameters
        ----------
        width : int
            The maximum width of the text.

        Returns
        -------
        str
            A text representation of the options table.
        """
        rst = self.__rst__()
        cols = [len(header) for header in rst[0].split()]
        desc_col = sum(cols[:-1]) + len(cols) - 1
        desc_len = width - desc_col

        # if it won't fit in allowed width, just return the rST
        if desc_len < 10:
            return "\n".join(rst)

        text = []
        for row in rst:
            if len(row) > width:
                text.append(row[:width])
                if not row.startswith("==="):
                    row = row[width:].rstrip()
                    while len(row) > 0:
                        text.append(" " * desc_col + row[:desc_len])
                        row = row[desc_len:]
            else:
                text.append(row)

        return "\n".join(text)

    def _assert_valid(self, name, value):
        """
        Check whether the given value is valid, where the key has already been declared.

        The optional checks consist of ensuring: the value is one of a list of acceptable values,
        the type of value is one of a list of acceptable types, value is not less than lower,
        value is not greater than upper, and value satisfies check_valid.

        Parameters
        ----------
        name : str
            The key for the declared option.
        value : object
            The default or user-set value to check for value, type, lower, and upper.
        """
        meta = self._dict[name]
        values = meta["values"]
        dtype = meta["dtype"]
        lower = meta["lower"]
        upper = meta["upper"]

        if not (value is None and meta["allow_none"]):
            # If only values is declared
            if values is not None:
                if value not in values:
                    raise ValueError(
                        f"Value ({value!r}) of option {name!r} is not one of {values}."
                    )
            # If only dtype is declared
            elif dtype is not None:
                if not isinstance(value, dtype):
                    vtype = type(value)
                    raise TypeError(
                        "Value ({!r}) of option {!r} has type of ({}), but "
                        "expected type ({}).".format(value, name, vtype, dtype)
                    )

            if upper is not None:
                if value > upper:
                    raise ValueError(
                        f"Value ({value!r}) of option {name!r} exceeds maximum allowed value {upper}."
                    )
            if lower is not None:
                if value < lower:
                    raise ValueError(
                        f"Value ({value!r}) of option {name!r} is less than minimum allowed value {lower}."
                    )

        # General function test
        if meta["check_valid"] is not None:
            meta["check_valid"](name, value)

    def declare(
        self,
        name,
        default=_undefined,
        values=None,
        dtype=None,
        desc="",
        upper=None,
        lower=None,
        check_valid=None,
        allow_none=False,
        tag=None,
    ):
        r"""
        Declare an option.

        The value of the option must satisfy the following:
        1. If values only was given when declaring, value must be in values.
        2. If dtype only was given when declaring, value must satisfy isinstance(value, dtype).
        3. It is an error if both values and dtype are given.

        Parameters
        ----------
        name : str
            Name of the option.
        default : object or Null
            Optional default value that must be valid under the above 3 conditions.
        values : set or list or tuple or None
            Optional list of acceptable option values.
        dtype : type or tuple of types or None
            Optional type or list of acceptable optional types.
        desc : str
            Optional description of the option.
        upper : float or None
            Maximum allowable value.
        lower : float or None
            Minimum allowable value.
        check_valid : function or None
            General check function that raises an exception if value is not valid.
        allow_none : bool
            If True, allow None as a value regardless of values or dtype.
        tag : Optional[str]
            Specify a tag to allow later filtering of options.
        """

        if values is not None and not isinstance(values, (set, list, tuple)):
            raise TypeError(
                "In declaration of option '%s', the 'values' arg must be of type None,"
                " list, or tuple - not %s." % (name, values)
            )
        if dtype is not None and not isinstance(dtype, (type, set, list, tuple)):
            raise TypeError(
                "In declaration of option '%s', the 'dtype' arg must be None, a type "
                "or a tuple - not %s." % (name, dtype)
            )

        if dtype is not None and values is not None:
            raise RuntimeError(
                "'dtype' and 'values' were both specified for option '%s'." % name
            )

        default_provided = default is not _undefined
        default_value = default() if isinstance(default, Callable) else default

        self._dict[name] = {
            "value": default_value,
            "values": values,
            "dtype": dtype,
            "desc": desc,
            "upper": upper,
            "lower": lower,
            "check_valid": check_valid,
            "has_been_set": default_provided,
            "allow_none": allow_none,
            "tag": tag,
        }

        # If a default is given, check for validity
        if default_provided:
            self._assert_valid(name, default_value)

    def undeclare(self, name):
        """
        Remove entry from the OptionsDictionary, for classes that don't use that option.

        Parameters
        ----------
        name : str
            The name of a key, the entry of which will be removed from the internal dictionary.

        """
        if name in self._dict:
            del self._dict[name]

    def update(self, in_dict):
        """
        Update the internal dictionary with the given one.

        Parameters
        ----------
        in_dict : dict
            The incoming dictionary to add to the internal one.
        """
        for name in in_dict:
            self[name] = in_dict[name]

    def clear(self):
        if self._read_only:
            raise KeyError(f"Cannot clear read-only dictionnary {self}.")
        self._dict.clear()

    def __iter__(self):
        """
        Provide an iterator.

        Returns
        -------
        iterable
            iterator over the keys in the dictionary.
        """
        return iter(self._dict)

    def keys(self):
        """Return an iterator over dict keys."""
        return self._dict.keys()

    def values(self):
        """Return an iterator over dict values.
        Raises `RuntimeError` if a required option is undefined.
        """
        for key in self.keys():
            yield self[key]

    def items(self, tags: Container[str] = ()):
        """Return an iterator over dict keys and values, as tuples.

        The items can be filtered by tags if provided.

        Raises `RuntimeError` if a required option is undefined.

        Parameters
        ----------
        tags : Container[str]
            tags to filter

        Returns
        -------
        items
            iterator to the filtered items.
        """

        def get_item(x):
            key, val = x
            return key, self[key]

        def filter_items(x):
            _, val = x
            return not tags or val["tag"] in tags

        return map(get_item, filter(filter_items, self._dict.items()))

    def __contains__(self, key):
        """
        Check if the key is in the local dictionary.

        Parameters
        ----------
        key : str
            name of the option.

        Returns
        -------
        boolean
            whether key is in the local dict.
        """
        return key in self._dict

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, name: str, value):
        """
        Set an option in the local dictionary.

        Parameters
        ----------
        name : str
            name of the option.
        value : -
            value of the option to be value- and type-checked if declared.
        """
        if self._read_only:
            raise KeyError(f"Tried to set read-only option {name!r}.")

        try:
            meta = self._dict[name]
        except KeyError:
            # The key must have been declared.
            msg = f"Option {name!r} cannot be set because it has not been declared."
            raise KeyError(msg)

        self._assert_valid(name, value)

        meta["value"] = value
        meta["has_been_set"] = True

    def __getitem__(self, name):
        """
        Get an option from the dict or declared default.

        Parameters
        ----------
        name : str
            name of the option.

        Returns
        -------
        value : -
            value of the option.
        """
        # If the option has been set in this system, return the set value
        try:
            meta = self._dict[name]
            if meta["has_been_set"]:
                return meta["value"]
            else:
                raise RuntimeError(f"Option {name!r} is required but has not been set.")
        except KeyError:
            raise KeyError(f"Option {name!r} cannot be found")

    def __getattr__(self, name):
        if name in self.__getattribute__("_dict"):
            return self[name]
        return self.__getattribute__(name)

    def get(self, name: str, default: Optional[Any] = None) -> Any:
        if name in self:
            return self[name]

        return default

    def merge(self, other: OptionsDictionary) -> None:
        self._dict.update(other._dict)


class HasOptions:
    """Base class to handle options as `OptionsDictionary`."""

    __slots__ = ()  # do not provide the '_options' slot to allow multiple inheritence

    def __init__(self):
        self._options = OptionsDictionary()

    def _declare_options(self) -> None:
        """Declares options."""
        pass

    def _init_options(self, kwargs) -> OptionsDictionary:
        """Initializes the options."""
        self._declare_options()
        self._consume_options(kwargs)
        self._set_options()

        return self._options

    def _consume_options(self, kwargs: Dict[str, Any]) -> None:
        """Sets the options by consuming them."""
        for key, _ in self._options.items():
            if key in kwargs:
                self._options[key] = kwargs.pop(key)

    def _set_options(self) -> None:
        """Sets options from the current state."""
        pass

    def set_options(self) -> None:
        """Sets options from the current state."""
        self._set_options()


class HasCompositeOptions(HasOptions):

    __slots__ = ()

    def _get_nested_objects_with_options(self) -> Iterable[HasOptions]:
        """Gets nested objects having options."""
        return ()

    def _init_options(self, kwargs) -> OptionsDictionary:
        """Initializes the options."""
        super()._init_options(kwargs)

        for obj in self._get_nested_objects_with_options():
            obj._init_options(kwargs)
            self._options.merge(obj._options)

        return self._options

    def set_options(self) -> None:
        """Sets options from the current state."""
        self._set_options()
        for obj in self._get_nested_objects_with_options():
            obj.set_options()
