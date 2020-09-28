from typing import Any, Mapping, NamedTuple, Union


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
    mapping : Mapping[str, Any] or ExtensiblePort
        Map object with string key
    """
    context: "System"
    mapping: "Union[Mapping[str, Any], ExtensiblePort]"
    key: str
    
    def __repr__(self) -> str:
        name = self.context.name
        mapping_name = getattr(self.mapping, "name", "")
        if mapping_name:
            name = ".".join((name, mapping_name))
        return f'<VariableReference ({name}, {self.key})>'
