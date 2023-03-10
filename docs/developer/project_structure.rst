Project Code Structure
----------------------

CoSApp_ is a Python library, with sporadic JavaScript and HTML content intended for object rendering in Jupyter notebooks.
It allows the creation of elemental simulation units referred to as *systems*, and the assembly of existing systems into composite models.
More importantly, it is focused on the design of such systems, that is the computation of critical parameters to satisfy operating constraints.

Package description
~~~~~~~~~~~~~~~~~~~~

Users define their models by specializing base classes contained in module :py:mod:`cosapp.base`.
In particular:

- :py:class:`~cosapp.base.System`, defining systems;
- :py:class:`~cosapp.base.Port`, defining variable sets at the interface of systems;
- :py:class:`~cosapp.base.Driver`, defining algorithms acting on systems, referred to as *drivers*.

Drivers typically modify system variables to solve mathematical problems
addressing specific simulation intents. For example:

- :py:class:`~cosapp.drivers.NonLinearSolver`, to solve design problems and/or equilibrate systems with loops;
- :py:class:`~cosapp.drivers.RungeKutta`, for transient simulations with a Runge-Kutta explicit scheme;
- :py:class:`~cosapp.drivers.Optimizer`, to minimize or maximize an objective in the system.

Various helpers are also available:

- :py:mod:`~cosapp.recorders`: probes recording simulation results;
- :py:func:`~cosapp.utils.set_log`, to activate a simulation logger;
- :py:func:`~cosapp.tools.to_fmu`, to export a CoSApp system as an FMU;
- :py:func:`~cosapp.tools.display_doc`, to display the documentation of CoSApp .

:py:mod:`cosapp` itself is a Python `namespace <https://docs.python.org/3/reference/import.html#namespace-packages>`_
rather than a traditional Python package. This allows third-party packages to be included under the same umbrella.
