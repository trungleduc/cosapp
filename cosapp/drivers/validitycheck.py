import logging
from typing import NoReturn, Union

from cosapp.utils.validate import validate
from cosapp.drivers.optionaldriver import OptionalDriver

logger = logging.getLogger(__name__)


# TODO
# [ ] Quid for vector variables
class ValidityCheck(OptionalDriver):
    """
    When executed, this driver reports in the log the validity status for all variables of the
    driver `System` owner (and its children).

    Parameters
    ----------
    name : str
        Name of the driver
    owner : System, optional
        :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
    **kwargs : Any
        Keyword arguments will be used to set driver options
    """
    
    __slots__ = ('warnings', 'errors', )

    def __init__(
        self, name: str, owner: "Optional[cosapp.systems.System]" = None, **kwargs
    ) -> NoReturn:
        """Initialize a driver

        Parameters
        ----------
        name: str, optional
            Name of the `Module`
        owner : System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belong; default None
        **kwargs : Dict[str, Any]
            Optional keywords arguments
        """
        super().__init__(name, owner, **kwargs)
        self.warnings = dict()  # type: Dict[str, str]
        self.errors = dict()  # type: Dict[str, str]

    def compute(self) -> NoReturn:
        """Report in the log the validity status for all variables of the driver `System` owner
        (and its children).
        """
        def message(log_dict):
            return "\n" + "\n\t".join(["{}{}".format(key, msg) for key, msg in log_dict.items()])
        self.warnings, self.errors = validate(self.owner)

        if self.warnings:
            logger.warning(message(self.warnings))

        if self.errors:
            logger.error(message(self.errors))
