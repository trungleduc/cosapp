from typing import Any, Mapping, NamedTuple, Union
from cosapp.utils.naming import natural_varname


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
    context: "System"
    mapping: "Union[Mapping[str, Any], BasePort]"
    key: str

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
        return f"<VariableReference ({name}, {self.key})>"

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
