from __future__ import annotations
from typing import Any, Mapping, NamedTuple, Union, Optional, TYPE_CHECKING
from cosapp.utils.naming import natural_varname
import numpy as np
if TYPE_CHECKING:
    from cosapp.systems import System
    from cosapp.ports.port import BasePort


# The real storage of a value is to its final port. Therefore the name mapping, to speed-up the
# getter and setter, must returns directly the object from its direct dictionary-like container.
class VariableReference(NamedTuple):
    """Tuple allowing to retrieve or set a variable from a map with string 
    key.

    Attributes
    ----------
    context : System
        Context of the variable
    key : str
        Key of the variable in the dictionary
    mapping : Mapping[str, Any] or BasePort
        Map object with string key
    """
    context: System
    mapping: Union[Mapping[str, Any], BasePort]
    key: str
    mask: Optional[np.ndarray] = None

    @property
    def value(self) -> Any:
        """Any: variable reference value"""
        return self.mapping[self.key]

    @value.setter
    def value(self, value: Any) -> None:
        self.mapping[self.key] = value
    
    def __repr__(self) -> str:
        name = self.context.name
        mapping_name = self._mapping_name()
        if mapping_name:
            name = f"{name}.{mapping_name}"
        return f"<{self.__class__.__name__} ({name}, {self.key})>"

    @property
    def name(self) -> str:
        mapping_name = self._mapping_name()
        if mapping_name:
            return natural_varname(f"{mapping_name}.{self.key}")
        return self.key

    @property
    def contextual_name(self) -> str:
        return natural_varname(f"{self.context.name}.{self.name}")
    
    def _mapping_name(self) -> str:
        return getattr(self.mapping, "name", "")


class MaskedVariableReference(VariableReference):
    """Tuple allowing to retrieve or set a variable from a map with string 
    key.

    Attributes
    ----------
    context : System
        Context of the variable
    key : str
        Key of the variable in the dictionary
    mapping : Mapping[str, Any] or BasePort
        Map object with string key
    mask : Sequence[bool]
        Mask to be applied on the variable
    """
    mask: np.ndarray

    @property
    def value(self) -> Any:
        """Any: variable reference value"""
        print(self.mask)
        return self.mapping[self.key][self.mask]

    @value.setter
    def value(self, value: Any) -> None:
        self.mapping[self.key][self.mask] = value

    def __repr__(self) -> str:
        unmasked_name = super().__repr__()
        return f"{unmasked_name}[{self.mask}]"
    
    @property
    def name(self) -> str:
        mapping_name = self._mapping_name()
        if mapping_name:
            return natural_varname(f"{mapping_name}.{self.key}[{self.mask}]")
        return self.key

    @property
    def contextual_name(self) -> str:
        return natural_varname(f"{self.context.name}.{self.name}[{self.mask}]")
    
    def _mapping_name(self) -> str:
        return getattr(self.mapping, "name", "")
    
    def __copy__(self):
        return MaskedVariableReference(self.context, self.mapping, self.key, self.mask.copy())