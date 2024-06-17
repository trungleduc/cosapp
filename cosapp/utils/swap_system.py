# Copyright (C) 2023 twiinIT - All Rights Reserved
# Licensed under the Apache License, Version 2.0

from __future__ import annotations
import copy
import logging
from typing import TYPE_CHECKING
from cosapp.utils.helpers import check_arg
if TYPE_CHECKING:
    from cosapp.base import System

logger = logging.getLogger(__name__)


def swap_system(old_system: System, new_system: System, init_values=True):
    """Replace `old_system` by `new_system`.

    Parameters
    ----------
    - old_system [System]: System to replace
    - new_system [System]: Replacement system
    - init_values [bool, optional]: If `True` (default), original
        system values are copied into the replacement system.
    
    Returns
    -------
    old_system [System]: the original system, devoid of parent.
    """
    from cosapp.base import System
    from cosapp.ports.port import BasePort

    check_arg(old_system, "old_system", System)
    check_arg(new_system, "new_system", System)

    if new_system.parent is not None:
        raise ValueError(
            f"System {new_system.full_name()!r} already belongs to a system tree."
        )
    if old_system.parent is None:
        raise ValueError(
            f"Cannot replace top system {old_system.full_name()!r}."
        )
    if new_system.name != old_system.name:
        logger.info(
            f"New system {new_system.name!r} renamed into {old_system.name!r} inside {old_system.parent.full_name()!r}"
        )

    new_system.parent = parent = old_system.parent
    new_system.name = system_name = old_system.name

    # connections list
    to_restore = list(
        filter(
            lambda c: (c.source.owner is old_system) or (c.sink.owner is old_system),
            parent.all_connectors(),
        )
    )

    # update child in parent
    execution_idx = list(parent.exec_order).index(system_name)
    parent.pop_child(system_name)
    parent.add_child(new_system, execution_index=execution_idx)

    # restore connections
    for connector in to_restore:
        sink = connector.sink
        source = connector.source
        if sink.owner is old_system:
            connector.sink = new_system[sink.name]
        if source.owner is old_system:
            connector.source = new_system[source.name]
        parent.connect(connector.sink, connector.source, connector.mapping)

    # init values
    if init_values:
        not_copied = set()
        for name, old_variable in old_system.name2variable.items():
            if not isinstance(old_variable.mapping, BasePort):
                continue
            try:
                new_variable = new_system.name2variable[name]
            except KeyError:
                not_copied.add(old_variable.contextual_name)
                continue
            try:
                new_variable.value = copy.deepcopy(old_variable.value)
            except:
                not_copied.add(old_variable.contextual_name)

        if not_copied:
            logger.warning(
                f"Could not copy {sorted(not_copied)} into {new_system.full_name()!r}"
            )

    return old_system
