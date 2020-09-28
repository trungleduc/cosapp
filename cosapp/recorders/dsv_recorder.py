"""Recorder to Delimited Separated Value format file."""
import os
from numbers import Number
from typing import Any, List, Optional, Tuple, Union
from collections.abc import Collection

import numpy
import pandas

from cosapp.recorders.recorder import BaseRecorder
from cosapp.utils.helpers import is_numerical, check_arg


class DSVRecorder(BaseRecorder):
    """Record data into Delimiter Separated Value file.

    Matching pattern are case sensitive and support the following special patterns:

    ========  ================================
    Pattern   Meaning
    ========  ================================
    `*`       matches everything
    `?`       matches any single character
    `[seq]`   matches any character in seq
    `[!seq]`  matches any character not in seq
    ========  ================================

    Excluding pattern are shadowing includes one; e.g. if `includes='*port_in.*' and `excludes='*.Pt'`, for a port
    having variables named `Tt` and `Pt`, only `Tt` will be recorded.

    Parameters
    ----------
    filepath : str
        Filepath to save data into.
    delimiter : `','`, `';'` or `'\t'`, optional
        Delimiter of data in the file; default `','`.
    buffer : bool, optional
        Should the data written after the simulation (`False`) or every time they are available (`True`);
        default `False`.
    includes : str or list of str, optional
        Variables matching these patterns will be included; default `'*'` (i.e. all variables).
    excludes : str or list of str or None, optional
        Variables matching these patterns will be excluded; default `None` (i.e. nothing is excluded).
    numerical_only : bool, optional
        Keep only numerical variables (i.e. number or numerical vector); default False.
    section : str, optional
        Current section name; default `''`.
    precision : int, optional
        Precision digits when writing floating point number; default 9 (i.e. 10 figures will be written).
    hold : bool, optional
        Append the new data or not; default `False`.
    raw_output : bool, optional
        Raw output; default `False`.

    ..note::
        Do not mention `inwards` or `outwards` in `includes` or `excludes` list. Otherwise you may not record the wanted
        variables.
    """

    def __init__(
        self,
        filepath: str,
        delimiter: str = ",",
        buffer: bool = False,
        includes: Union[str, List[str]] = "*",
        excludes: Optional[Union[str, List[str]]] = None,
        numerical_only: bool = False,
        section: str = "",
        precision: int = 9,
        hold: bool = False,
        raw_output: bool = False,
    ):
        check_arg(filepath, "filepath", str)
        check_arg(delimiter, "delimiter", str)
        check_arg(buffer, "buffer", bool)

        supported_delimiters = (",", ";", "\t")
        if delimiter not in supported_delimiters:
            raise ValueError(
                f"Supported delimiters are {supported_delimiters}; got {delimiter!r}"
            )

        super().__init__(
            includes, excludes, numerical_only, section, precision, hold, raw_output
        )

        self._filepath = filepath  # type: str
        self._delimiter = delimiter  # type: str
        self.__tmp = list() if buffer else None  # type: List[List[Any]]

    @property
    def data(self) -> pandas.DataFrame:
        """pandas.DataFrame : DataFrame with the results."""

        if os.path.exists(self._filepath):
            if self.__tmp is not None:
                with open(self._filepath, "r") as f:
                    headers = f.readline().split(self._delimiter)
                return pandas.DataFrame(self.__tmp, columns=headers)
            else:
                return pandas.read_csv(self._filepath, delimiter=self._delimiter, header=0)
        else:
            return pandas.DataFrame()

    @property
    def _raw_data(self) -> List[List[Any]]:
        """Return a raw/unformatted version of the records.

        Returns
        -------
        List[List[Any]]
            The records of the `watched_object` for the variables given by the `get_variables_list` method
        """
        if not os.path.exists(self._filepath):
            return list()

        if self.__tmp is not None:
            return self.__tmp

        with open(self._filepath, "r") as f:
            # The header line is skipped
            content = map(lambda line: line.split(self._delimiter), f.readlines()[1:])
        return content

    def start(self):
        """Initialize recording support."""
        super().start()
        if not self.hold:
            # TODO Fred we could use clean/dirty here
            self.watched_object.run_once()  # Run the System to be sure the list are developed.
            if self.__tmp is not None:
                self.__tmp.clear()

            headers = [
                self.SPECIALS.section,
                self.SPECIALS.status,
                self.SPECIALS.code,
                self.SPECIALS.reference,
            ]

            def append(lst, base, unit):
                if unit and not self._raw_output:
                    lst.append(" ".join((base, "[{}]".format(unit))))
                else:
                    lst.append(base)

            for name, unit in zip(
                self.get_variables_list(), self._get_units(self.get_variables_list())
            ):
                value = self.watched_object[name]
                unit = "" if self._raw_output else unit
                if isinstance(value, numpy.ndarray) and value.ndim > 0:
                    for i in range(value.size):
                        entry = "{}[{}]".format(name, i)
                        append(headers, entry, unit)
                elif isinstance(value, Collection) and not isinstance(value, str):
                    for i in range(len(value)):
                        entry = "{}[{}]".format(name, i)
                        append(headers, entry, unit)
                else:
                    append(headers, name, unit)

            with open(self._filepath, "w") as f:
                # Write header
                f.write(self._delimiter.join(headers) + "\n")

    def record_state(
        self, time_ref: Union[float, str], status: str = "", error_code: str = "0"
    ):
        """Record the watched object at the provided status.

        Parameters
        ----------
        time_ref : float or str
            Current simulation time (float) or point reference (str)
        status : str, optional
            Status of the simulation; default `''`.
        error_code : str, optional
            Error code; default `'0'`.
        """
        if self.paused:
            return

        line = list()

        line.append(self.section)
        line.append(status)
        line.append(error_code)
        line.append(str(time_ref))

        precision = self.precision

        def fmt(value, check=True):
            if check and not is_numerical(value):
                return str(value)
            return "{0:.{1}e}".format(value, precision)

        for name in self.get_variables_list():
            value = self.watched_object[name]
            if isinstance(value, numpy.ndarray):
                if numpy.issubdtype(value.dtype, numpy.number):
                    line.extend([fmt(v, check=False) for v in value.flatten()])
                else:
                    line.extend([str(v) for v in value.flatten()])
            elif isinstance(value, Collection) and not isinstance(value, str):
                line.extend([fmt(v) for v in value])
            else:
                line.append(fmt(value))

        if self.__tmp is None:
            with open(self._filepath, "a") as f:
                f.write(self._delimiter.join(line) + "\n")
        else:
            self.__tmp.append(line)

        super().record_state(time_ref=time_ref, status=status, error_code=error_code)

    def clear(self):
        """Clear all previously stored data."""
        if self.__tmp is not None:
            self.__tmp.clear()
        super().clear()

    def exit(self):
        """Close recording session."""
        if self.__tmp is not None:
            with open(self._filepath, "a") as f:
                for line in self.__tmp:
                    f.write(self._delimiter.join(line) + "\n")
