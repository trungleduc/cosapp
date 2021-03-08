History
=======

0.11.4 (2021-03-08)
---------------------

New feature:

* Recorders:
  It is now possible to add evaluable expressions in recorders (MR `#27 <https://gitlab.com/cosapp/cosapp/-/merge_requests/27>`_):
  
  .. code:: python

      point = PointMass('point')
      driver = point.add_driver(RungeKutta(order=3, time_interval=(0, 2), dt=0.01))

      recorder = driver.add_recorder(recorders.DataFrameRecorder(
          includes=['x', 'a', 'norm(v)']),  # norm(v) will be recorded in DataFrame
          period=0.1,
      )

Documentation

* New tutorial on `SystemSurrogate` (MR `#15 <https://gitlab.com/cosapp/cosapp/-/merge_requests/15>`_).

Bug fixes, minor improvements and code quality:

* Initialization bug in time simulations (MR `#23 <https://gitlab.com/cosapp/cosapp/-/merge_requests/23>`_).
* Bug in nonlinearity estimation in `NumericalSolver` (MR `#22 <https://gitlab.com/cosapp/cosapp/-/merge_requests/22>`_).
* Do not raise `ArithmeticError` when an unknown is declared several time (MR `#18 <https://gitlab.com/cosapp/cosapp/-/merge_requests/18>`_).
* Suppress deprecation warnings raised by `numpy` (MR `#20 <https://gitlab.com/cosapp/cosapp/-/merge_requests/20>`_ and `#24 <https://gitlab.com/cosapp/cosapp/-/merge_requests/24>`_).
* Suppress undue warning raised by `numpy` in `NonLinearSolver` (MR `#19 <https://gitlab.com/cosapp/cosapp/-/merge_requests/19>`_).
* Fix incompatibility between `pandas` and `xlrd` (MR `#21 <https://gitlab.com/cosapp/cosapp/-/merge_requests/21>`_).
* Other code quality improvement (MR `#16 <https://gitlab.com/cosapp/cosapp/-/merge_requests/16>`_, `#17 <https://gitlab.com/cosapp/cosapp/-/merge_requests/17>`_, `#26 <https://gitlab.com/cosapp/cosapp/-/merge_requests/26>`_, `#27 <https://gitlab.com/cosapp/cosapp/-/merge_requests/27>`_).

0.11.3 (2020-12-16)
---------------------

New feature:

* Surrogate models:
  It is now possible to create a surrogate model at any system level with new method `System.make_surrogate` (MR `#3 <https://gitlab.com/cosapp/cosapp/-/merge_requests/3>`_ and `#12 <https://gitlab.com/cosapp/cosapp/-/merge_requests/12>`_):
  
  .. code:: python

      plane = Aeroplane('plane')  # system with subsystems engine1 and engine2

      # Say engine systems have one input parameter `fuel_rate`
      # and possibly several outputs, and many sub-systems

      # Create training schedule for input data
      doe = pandas.DataFrame(
        # loads of input data
        columns=['fuel_rate', 'fan.diameter', ..]  # input names
      )
      plane.engine1.make_surrogate(doe)  # generates output data and train model

      plane.run_once()  # executes the surrogate model of `engine1` instead of original compute()
      
      # dump model to file
      plane.engine1.dump_surrogate('engine.bin')
      # load model into `engine2`:
      plane.engine2.load_surrogate('engine.bin')

      # deactivate surrogate model on demand
      plane.engine1.active_surrogate = plane.engine2.active_surrogate = False

Bug fixes, minor improvements and code quality:

* Add several US-common unit conversions (MR `#2 <https://gitlab.com/cosapp/cosapp/-/merge_requests/2>`_).
* New method to export cosapp system structure into a dictionary (MR `#5 <https://gitlab.com/cosapp/cosapp/-/merge_requests/5>`_)
* Make recorders capture port and system properties (MR `#8 <https://gitlab.com/cosapp/cosapp/-/merge_requests/8>`_).
* Fix Module/System naming bug: *'inwards' and 'outwards' are allowed as Module/System names* (MR `#9 <https://gitlab.com/cosapp/cosapp/-/merge_requests/9>`_).
* Broad code quality improvement (MR `#11 <https://gitlab.com/cosapp/cosapp/-/merge_requests/11>`_).

  * Replace `typing.NoReturn` by `None` when appropriate.
  * Rewording pass, typo and error fixes in tutorial notebooks.
  * Suppress a `DeprecationWarning` raised by `numpy` in class `Variable`.
  * Reformat many strings as Python f-strings, for clarity.
  * Symplify many occurrences of `str.join()` for just two elements.

Global rewording of tutorial notebooks, including a few error fixes.

0.11.2 (2020-09-28)
---------------------

First open-source version.
No major code change; mostly updates of license files, URLs in docs, and CI scripts.

0.11.1 (2020-07-22)
---------------------

Feature:

* Add the possibility to set boundary condition of transient simulation from interpolate profile.
* Add the possibility to prescribe a maximum step for transient variables.

Bugs and code quality:

* Bug fix in ``RunOnce`` driver, preventing undue call to ``run_once`` method.
* Bug fix in AssignString: force copy for numpy arrays.
* New tutorial for advanced features of time simulations in CosApp.

0.11.0 (2020-05-12)
---------------------

Feature:

* Improve documentation at various places, add documentation about the cosapp packages structure and sequence diagram for transient simulation.
* Add a advanced logger feature for CoSApp simulations.
* Update FMU export to PythonFMU 0.6.0
* New method ``System.add_property`` allowing users to create read-only properties.

Bugs and code quality:

* Suppress deprecation warnings raised by external dependencies. 
* Fix bug in AssignString with arrays, `AssignString` of the kind `'x = [0, 1, 2]'` won't change variable `x` into an array of integers,  if `x` is declared as an array of floats.
* Fix ``TimeStackUnknown`` not able to stack transient variables defined on a children System or with partially pulled transient variable.
* Fix the bug related to the initialization of ``rate`` attributes in systems.

0.10.2 (2020-04-21)
-------------------

Feature:

* [BETA] Export CoSApp System as FMU

Bugs and code quality:

* Apply Broyden correction on Jacobian matrix for iteration without Jacobian update
* Support varying time step
* Fix time not being set before ``setup_run`` are called.
* Fix reference for residues in ``IterativeConnector`` (it equals 1. now)
* Drop pyhamcrest for pytest

0.10.1 (2020-01-15)
-------------------

Feature:

* Time varying boundary conditions are now possible

.. code:: python

    system = MySystem('something')  # system with transient variables x and v
    driver = system.add_driver(RungeKutta(time_interval=(0, 2), dt=0.01, order=3))
    
    driver.set_scenario(
        init = {'x': 0.5, 'v': 0},  # initial conditions
        values =
        {
            'omega': 0.7,
            'F_ext': '0.6 * cos(omega * t)'  # explicit time-dependency
        }
    )

Bugs and code quality:

* Fix various bug on the transient simulation front
* Correct implementation of step limitation in the Newton-Raphson solver
* Using a logger at ``DEBUG`` level will now display the call stack through the systems and drivers
* Rework of the Python evaluable string to be more efficient

0.10.0 (2019-10-23)
-------------------

* Introduce continuous time simulations with dedicated time drivers (see ``TimeDriver`` notebook in tutorials).
* Suppress notion of (un)freeze; all variables are considered as known, unless explicitly declared as unknowns.
* Drivers no longer use ports.
* Connectors are now stored by parent system.
* Migrate to pytest.

**API Changes:**

* Ports:

  * ``add_variable("x", units="m", types=Number)`` => ``add_variable("x", unit="m", dtype=Number)``
  * ``freeze`` => removed
  * ``unfreeze`` => replaced by ``add_unknown`` in Systems and Drivers
  * ``connect_to`` => replaced by ``connect`` at system level
  
* Systems:

  * ``time_ref`` is no longer an argument of method ``compute``:
  
    ``def compute(self, time_ref):`` => ``def compute(self):``
       
  * Create a new connection between ``a.in1`` and ``b.out``:
  
    ``self.a.in1.connect_to(self.b.out)`` => ``self.connect(self.a.in1, self.b.out)``
       
  * ``add_residues`` => ``add_equation``
  * ``set_numerical_default`` => Pass keyword to ``add_unknown``
  * ``add_inward("x", units="m", types=Number)`` => ``add_inward("x", unit="m", dtype=Number)``
  * ``add_outward("x", units="m", types=Number)`` => ``add_outward("x", unit="m", dtype=Number)``
  
* Drivers:

  * ``add_unknowns(maximal_absolute_step, maximal_relative_step, low_bound, high_bound)`` => ``add_unknown(max_abs_step, max_rel_step, lower_bound, upper_bound)``
  * ``add_equations`` => ``add_equation``
  * Equations are now represented by a unique string, instead of two strings (left-hand-side, right-hand-side):
  
    ``add_equations("a", "b")`` => ``add_equation("a == b")``  
    
    ``add_equations([("x", "2 * y + 1"), ("a", "b")])`` => ``add_equation(["x == 2 * y + 1", "a == b"])``  
        
  * For ``NonLinearSolver``:
  
    ``fatol`` and ``xtol`` => ``tol``  
    
    ``maxiter`` => ``max_iter``  
        
  * For ``Optimizer``:
  
    ``ftol`` => ``tol``
    
    ``maxiter`` => ``max_iter``

0.9.6 (2019-10-10)
------------------

* More correction for VISjs viewer and System HTML representation

0.9.5 (2019-09-25)
------------------

* Correct D3 & VISjs Viewers

0.9.4 (2019-09-25)
------------------

* Introduce an optional environment variable ``COSAPP_CONFIG_DIR``

0.9.3 (2019-07-25)
------------------

**! API Changes**

* MonteCarlo:

  * ``Montecarlo`` => ``MonteCarlo``
  * ``Montecarlo.add_input_vars`` => ``MonteCarlo.add_random_variable``
  * ``Montecarlo.add_response_vars`` => ``MonteCarlo.add_response``

* MonteCarlo has been improved by using Sobol random generator
* Viewers code on ``System`` is moved in a subpackage of ``cosapp.tools``
* Residue reference is now calculated only once
* Various bug fix

0.9.2 (2019-07-01)
------------------
* In nonlinear solver, store LU factorization of the Jacobian matrix, rather than its inverse.
* Minor refactoring of the core source code, with no API changes

0.9.1 (2019-04-23)
------------------

* Create ``Variable`` class to manage variable attributes
* ``watchdog`` is now optional
* Configuration is now inside a folder ``$HOME/.cosapp.d``
* API changes:
  - ``get_latest_solution`` => ``save_solution``
  - ``load_solver_solution`` => ``load_solution``
* Various bug fix

0.9.0 (2019-03-04)
------------------

This release introduces lots of API changes:

* Core ports and unit are available in ``cosapp.ports``
* Core systems are available in ``cosapp.systems``
* Core drivers are available in ``cosapp.drivers``
* Core recorders are available in ``cosapp.recorders``
* Core tools are available in ``cosapp.tools``
* Core notebook tools are available in ``cosapp.notebook`` (! this is now a separated package)
* ``data`` have been renamed in ``inwards`` and ``add_data`` in ``add_inward``
* ``locals`` have been renamed in ``outwards`` and ``add_locals`` in ``add_outward``
* ``BaseRecorder.record_iteration`` renamed in ``BaseRecorder.record_state``

- Huge code refractoring: cosapp is now a `Python namespace <https://packaging.python.org/guides/packaging-namespace-packages/>`_.
- ``cosapp.notebook`` has been moved to an independent package ``cosapp_notebook``. But it is still accessible from ``cosapp.notebook``.
- Introduce *Signal* / *Slot* pattern to connect to internal event (implementation from `signalslot <https://github.com/Numergy/signalslot>`_, included in ``cosapp.core.signal``)
    * ``Module.setup_ran``: Signal emitted after the ``call_setup_run`` execution
    * ``Module.computed``: Signal emitted after the full ``compute`` stack (i.e.: ``_postcompute``)
    * ``Module.clean_ran``: Signal emitted after the ``call_clean_run`` execution
    * ``BaseRecorder.state_recorded``: Signale emitted after the ``record_state`` execution

0.8.0 (2018-10-26)
------------------

- Add Jacobian partial matrix update
- Add numerical features to variables to ease convergence control
- Add monitoring of solver residues
- Add restoration of solver result for initialization
- Rework residues and unknowns handling (remove virtual port and pulling port)
- Rework optimizer to be more homogeneous with non-linear solver
- Improve linear Monte Carlo computation time
- Improve data viewer for non-linear solver
- Create viewer for Monte Carlo
- Add dropdown widget for enum variables

0.7.0 (2018-09-17)
------------------

- Add helper functions to present solver evolutions
- Add new d3 system visualization

0.6.0 (2018-08-14)
------------------

- Implement clean-dirty politic
- Restore compatibility with Python 3.4
- Display influence matrix

0.5.0 (2018-07-20)
------------------

- Simplify drivers structure, all actions for a case are supported by a single class ``RunSingleCase``
- Add support for vector variables; they can be partially (un)frozen and are handled correctly by the solver.
- Add ``MonteCarlo`` driver
- Add recording data capability

0.4.0 (2018-06-15)
------------------

- ``System`` and ``Driver`` have now a common ancestor ``Module`` => ``Driver`` variables are now stored as data or locals
- Add visualization of ``System`` connections based on N2 graph (syntax: ``cosapp.viewmodel(mySystem)``)

0.3.0 (2018-04-05)
------------------

API changes: ``System.add_driver`` and ``Driver.add_child`` take now an instance of ``Driver``

- Add external code caller System
- Add validation range attributes on variables
- Add variable visibility
- Add metamodel training and DoE generator
- Add helper function to list inputs and outputs variables of a ``System``

0.2.0 (2018-03-01)
------------------

* Stabilization of the user API

0.1.0 (2018-01-02)
------------------

* First release.
