"""Recorder in `pandas.DataFrame`."""
import copy
from numbers import Number
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cosapp.recorders.recorder import BaseRecorder


class DataFrameRecorder(BaseRecorder):
    """Record data into a pandas.DataFrame.

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
        includes: Union[str, List[str]] = "*",
        excludes: Optional[Union[str, List[str]]] = None,
        # metadata: Optional[Union[str, List[str]]] = None,
        numerical_only: bool = False,
        section: str = "",
        precision: int = 9,
        hold: bool = False,
        raw_output: bool = True,
    ):
        super().__init__(
            includes, excludes, numerical_only, section, precision, hold, raw_output)

        # Temporary storage
        self.__tmp = list()  # type: List[List[Any]]

    @property
    def data(self) -> pd.DataFrame:
        """pandas.DataFrame : DataFrame storing the results."""
        # According to DataFrame documentation, it is more efficient to store in a list then create the DataFrame
        headers = [
            self.SPECIALS.section,
            self.SPECIALS.status,
            self.SPECIALS.code,
            self.SPECIALS.reference,
        ]
        if self._raw_output:
            headers.extend(self.get_variables_list())
        else:
            headers.extend(
                map(
                    lambda v: "{} [{}]".format(v[0], v[1]),
                    zip(
                        self.get_variables_list(),
                        self._get_units(self.get_variables_list()),
                    ),
                )
            )

        return pd.DataFrame(self.__tmp, columns=headers)

    @property
    def _raw_data(self) -> List[List[Any]]:
        """Return a raw/unformatted version of the records

        Returns
        -------
        List[List[Any]]
            The records of the `watched_object` for the variables given by the `get_variables_list` method
        """
        return self.__tmp

    def start(self):
        """Initialize recording support."""
        super().start()
        if not self.hold:
            self.__tmp.clear()

    def record_state(self,
        time_ref: Union[float, str],
        status: str = "",
        error_code: str = "0"
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

        line = [self.section, status, error_code, str(time_ref)]
        for name in self.get_variables_list():
            if name in self.watched_object:
                line.append(copy.deepcopy(self.watched_object[name]))
            else:
                line.append(np.nan)

        self.__tmp.append(line)

        super().record_state(
            time_ref=time_ref, status=status, error_code=error_code)

    def exit(self):
        """Close recording session."""
        pass

    def clear(self):
        """Clear all previously stored data."""
        self.__tmp.clear()
        super().clear()
