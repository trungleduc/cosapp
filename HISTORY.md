# History

## 0.12.0 (2022-01-17)

### New features

* Implementation of multimode systems and hybrid continuous/discrete time solver (MRs [#100](https://gitlab.com/cosapp/cosapp/-/merge_requests/100), [#103](https://gitlab.com/cosapp/cosapp/-/merge_requests/103), [#105](https://gitlab.com/cosapp/cosapp/-/merge_requests/105)-[#108](https://gitlab.com/cosapp/cosapp/-/merge_requests/108), [#110](https://gitlab.com/cosapp/cosapp/-/merge_requests/110)-[#121](https://gitlab.com/cosapp/cosapp/-/merge_requests/121)):
  * Possibility to declare events and mode variables in systems.
  * New method `System.transition` describing system transition upon the occurrence of events.
  * Event detection in `ExplicitTimeDriver`.

* Possibility to specify a stop criterion in time simulation scenarios (MR [#107](https://gitlab.com/cosapp/cosapp/-/merge_requests/107)).

### Documentation

* New tutorial on hybrid time simulations and multimode systems (MRs [#116](https://gitlab.com/cosapp/cosapp/-/merge_requests/116) and [#120](https://gitlab.com/cosapp/cosapp/-/merge_requests/120)).
* Updated build config file following up a bug fix in [github.com/readthedocs](https://github.com/readthedocs/readthedocs.org) (MR [#109](https://gitlab.com/cosapp/cosapp/-/merge_requests/109)).

## 0.11.8 (2021-11-24)

### New features & API evolutions

* New module `cosapp.base` (MR [#96](https://gitlab.com/cosapp/cosapp/-/merge_requests/96)) containing base classes for user-defined classes (in particular, `Port`, `System` and `Driver`). Also contains `BaseConnector`, base class for custom connectors (see "User-defined and peer-to-peer connectors" below), as well as CoSApp-specific exceptions `ScopeError`, `UnitError` and `ConnectorError`.
**Note:** `Port`, `System` and `Driver` can still be imported from `cosapp.ports`, `cosapp.systems` and `cosapp.drivers`, respectively.

* Public API `cosapp.base.SurrogateModel` to define custom surrogate models used in `System.make_surrogate` (MR [#97](https://gitlab.com/cosapp/cosapp/-/merge_requests/97)).
Pre-defined models have been moved to module `cosapp.utils.surrogate_models`.

* System-to-system connections (MR [#94](https://gitlab.com/cosapp/cosapp/-/merge_requests/94)).

```python
class LegacyPortToPort(System):
    def setup(self):
        a = self.add_child(ModelA('a'))
        b = self.add_child(ModelB('b'))

        # Explicit port-to-port connections
        self.connect(a.p_in, b.p_out)
        self.connect(a.outwards, b.inwards, {'y': 'x'})

class Alternative(System):
  """Same as `LegacyPortToPort`, with alternative connection syntax"""
    def setup(self):
        a = self.add_child(ModelA('a'))
        b = self.add_child(ModelB('b'))

        # Alternative syntax: connect systems, with port or variable mapping
        self.connect(a, b, {'p_in': 'p_out', 'y': 'x'})
```

* User-defined and peer-to-peer connectors (MR [#87](https://gitlab.com/cosapp/cosapp/-/merge_requests/87) and MR [#98](https://gitlab.com/cosapp/cosapp/-/merge_requests/98)).

```python
import numpy
from copy import deepcopy
from cosapp.base import Port, System, BaseConnector

class DeepCopyConnector(BaseConnector):
  """User-defined deep-copy connector"""
    def transfer(self) -> None:
        source, sink = self.source, self.sink

        for target, origin in self.mapping.items():
            value = getattr(source, origin)
            setattr(sink, target, deepcopy(value))

class CustomPort(Port):
    def setup(self):
        self.add_variable('x', 0.0)
        self.add_variable('y', 1.0)

    class Connector(BaseConnector):
      """Connector for peer-to-peer connections"""
        def transfer(self) -> None:
            source, sink = self.source, self.sink
            sink.x = source.y
            sink.y = -source.x

class MyModel(System):
    def setup(self):
        self.add_input(CustomPort, 'p_in')
        self.add_output(CustomPort, 'p_out')

        self.add_inward('entry', numpy.identity(3))
        self.add_outward('exit', numpy.zeros_like(self.entry))

class Assembly(System):
    def setup(self):
        a = self.add_child(MyModel('a'))
        b = self.add_child(MyModel('b'))

        self.connect(a, b, {'exit', 'entry'}, cls=DeepCopyConnector)
        self.connect(a.p_in, b.p_out)  # will use CustomPort.Connector
```

### Documentation

* Updated tutorials (MR [#102](https://gitlab.com/cosapp/cosapp/-/merge_requests/102)).
  * Include latest API evolutions in tutorials on ports and systems.
  * New tutorial on user-defined connectors.
  * New section on user-defined surrogate models.

### Bug fixes and code quality

* Simplify loop resolution (MR [#77](https://gitlab.com/cosapp/cosapp/-/merge_requests/77), [#101](https://gitlab.com/cosapp/cosapp/-/merge_requests/101)).
* Improve connector transfer (MR [#99](https://gitlab.com/cosapp/cosapp/-/merge_requests/99)).
* Update binder settings, to take advantage of prebuilt plotly extension for jupyter lab (MR [#95](https://gitlab.com/cosapp/cosapp/-/merge_requests/95)).
* Other code quality improvements (MR [#76](https://gitlab.com/cosapp/cosapp/-/merge_requests/76), [#82](https://gitlab.com/cosapp/cosapp/-/merge_requests/82), [#90](https://gitlab.com/cosapp/cosapp/-/merge_requests/90)).

## 0.11.7 (2021-09-21)

### New features

* Possibility to define unknowns and equations at solver level (MR [#65](https://gitlab.com/cosapp/cosapp/-/merge_requests/65)).
Minor API evolution facilitating the definition of nonlinear problems and of multi-point design problems.

```python
engine = Turbofan('engine')
solver = engine.add_driver(NonLinearSolver('solver'))

# Add design points:
takeoff = solver.add_child(RunSingleCase('takeoff'))
cruise = solver.add_child(RunSingleCase('cruise'))

# Unknowns defined at solver level regarded as *design* unknowns
solver.add_unknown(['fan.diameter', 'core.turbine.inlet.area'])

# Local off-design equations can be directly defined at case level
takeoff.add_equation('thrust == 1.2e5')
cruise.add_equation('Mach == 0.8')
```

* New recursive iterator `tree()` for systems and drivers, yielding all elements in a composite tree (MR [#68](https://gitlab.com/cosapp/cosapp/-/merge_requests/68)).

```python
head = CompositeSystem('head')

bottom_to_top = [s.name for s in head.tree()]
top_to_bottom = [s.name for s in head.tree(downwards=True)]
```

* Visitor pattern for composite collections of systems, drivers and ports (MR [#68](https://gitlab.com/cosapp/cosapp/-/merge_requests/68)).

```python
from cosapp.patterns.visitor import Visitor, send as send_visitor

class DataCollector(Visitor):
    def __init__(self):
        self.data = {}

    def visit_system(self, system):
        key = system.full_name()
        self.data.setdefault(key, {})
        self.data[key]['children'] = [
            child.name for child in system.children.values()
        ]
        send_visitor(self, system.inputs.values())

    def visit_port(self, port):
        # specify what to do with a port

    def visit_driver(self, driver):
        # specify what to do with a driver

head = CompositeSystem('head')
collector = DataCollector()

send_visitor(collector, head.tree())
print(collector.data)
```

### Documentation

* New tutorials, and new "Tips & Tricks" notebook (MR [#71](https://gitlab.com/cosapp/cosapp/-/merge_requests/71)).

### Bug fixes and code quality

* Improved tests on clean/dirty status (MR [#66](https://gitlab.com/cosapp/cosapp/-/merge_requests/66)).
* Bug fix in tutorial notebook on validation (MR [#67](https://gitlab.com/cosapp/cosapp/-/merge_requests/67)).
* Code quality improvements (MR [#69](https://gitlab.com/cosapp/cosapp/-/merge_requests/69)).
* Make `System.exec_order` a view on `System.children` dictionary keys, rather than an independent attribute (MR [#70](https://gitlab.com/cosapp/cosapp/-/merge_requests/70)). Execution order can still be specified, via a dedicated setter for `exec_order`.
* Fix solver bugs occurring when system structure changes (MR [#73](https://gitlab.com/cosapp/cosapp/-/merge_requests/73)).

## 0.11.6 (2021-06-25)

### Bug fixes and code quality

* Resolve unknown aliasing for pulled input variables (MR [#58](https://gitlab.com/cosapp/cosapp/-/merge_requests/58)).
* Resolve bugs and issues related to `add_target` (MR [#61](https://gitlab.com/cosapp/cosapp/-/merge_requests/61)).
* Fix wrong comparison of Jacobian matrix `jac is None` after converting it as a `numpy` array (MR [#56](https://gitlab.com/cosapp/cosapp/-/merge_requests/56)).
* Set transparent background in `PortMarkdownFormatter` (MR [#59](https://gitlab.com/cosapp/cosapp/-/merge_requests/59)).
* Other code quality improvements (MR [#55](https://gitlab.com/cosapp/cosapp/-/merge_requests/55), [#57](https://gitlab.com/cosapp/cosapp/-/merge_requests/57), [#60](https://gitlab.com/cosapp/cosapp/-/merge_requests/60)).

## 0.11.5 (2021-05-07)

### New features

* **New [binder container](https://mybinder.org/v2/gl/cosapp%2Fcosapp/master?urlpath=lab/tree/docs/tutorials)**,
  allowing anyone to run interactively the tutorials used in the online documentation 
  (MR [#30](https://gitlab.com/cosapp/cosapp/-/merge_requests/30) and [#36](https://gitlab.com/cosapp/cosapp/-/merge_requests/36)).

* **Deferred equations to set targets:**

  New method `add_target`, defining a deferred equation on on a target variable (MR [#48](https://gitlab.com/cosapp/cosapp/-/merge_requests/48)).
  In effect, `add_target` creates an equation whose right-hand side is evaluated dynamically prior to each execution of the nonlienar solver.

  In the example below, the feature is illustrated in design mode. Outward `z` is a function of two independent variables `x` and `y`.
  When design method `'target_z'` is activated, the actual value of `z`, set interactively, is used as a target value, with unknown `y`:
  
```python
class SystemWithTarget(System):
    def setup(self):
        self.add_inward('x', 1.0)
        self.add_inward('y', 1.0)
        self.add_outward('z', 1.0)

        # Define design problem with a target on `z`
        design = self.add_design_method('target_z')
        design.add_unknown('y').add_target('z')

    def compute(self):
        self.z = self.x * self.y**2

s = SystemWithTarget('s')

solver = s.add_driver(NonLinearSolver('solver', tol=1e-9))
# Activate design method 'target_z': outward `z` becomes a target
solver.runner.design.extend(s.design_methods['target_z'])

s.x = 0.5
s.y = 0.5
s.z = 2.0  # set target
s.run_drivers()
assert s.y == pytest.approx(2)  # solution of x * y**2 == 2
assert s.z == pytest.approx(2)

s.z = 4.0  # dynamically set new target
s.run_drivers()
assert s.y == pytest.approx(np.sqrt(8))  # solution of x * y**2 == 4
assert s.z == pytest.approx(4)
```
  Targets can be also be declared in off-design mode, by calling `self.add_target(...)` in `System.setup`.

### Documentation

* Updated tutorials on `System`, `Driver`, and design methods (MR [#41](https://gitlab.com/cosapp/cosapp/-/merge_requests/41)).
* Typo fix and improvements in time driver tutorial (MR [#33](https://gitlab.com/cosapp/cosapp/-/merge_requests/33))

### Bug fixes, minor improvements and code quality

* Report variable name in unit-related Connector warning message (MR [#32](https://gitlab.com/cosapp/cosapp/-/merge_requests/32)).
* Automatically include field `time` in `DataFrame` recorders attached to a time driver (MR [#34](https://gitlab.com/cosapp/cosapp/-/merge_requests/34)).
* Bug fix in `MonteCarlo` driver (MR [#37](https://gitlab.com/cosapp/cosapp/-/merge_requests/37)).
* Replace `conda` by `mamba` in CI scripts (MR [#39](https://gitlab.com/cosapp/cosapp/-/merge_requests/39)).
* Revamp markdown and JS rendering of systems and ports (MR [#40](https://gitlab.com/cosapp/cosapp/-/merge_requests/40) and [#43](https://gitlab.com/cosapp/cosapp/-/merge_requests/43), [#47](https://gitlab.com/cosapp/cosapp/-/merge_requests/47) and [#51](https://gitlab.com/cosapp/cosapp/-/merge_requests/51)).
* Fix bug in `rate` type inference (MR [#44](https://gitlab.com/cosapp/cosapp/-/merge_requests/44)).
* Other code quality improvement (MR [#35](https://gitlab.com/cosapp/cosapp/-/merge_requests/35), [#38](https://gitlab.com/cosapp/cosapp/-/merge_requests/38), [#42](https://gitlab.com/cosapp/cosapp/-/merge_requests/42), [#46](https://gitlab.com/cosapp/cosapp/-/merge_requests/46), [#50](https://gitlab.com/cosapp/cosapp/-/merge_requests/50), [#52](https://gitlab.com/cosapp/cosapp/-/merge_requests/52), [#53](https://gitlab.com/cosapp/cosapp/-/merge_requests/53)).

## 0.11.4 (2021-03-08)

### New features

* Recorders:
  It is now possible to add evaluable expressions in recorders (MR [#27](https://gitlab.com/cosapp/cosapp/-/merge_requests/27)):
  
```python
point = PointMass('point')
driver = point.add_driver(RungeKutta(order=3, time_interval=(0, 2), dt=0.01))

recorder = driver.add_recorder(recorders.DataFrameRecorder(
    includes=['x', 'a', 'norm(v)']),  # norm(v) will be recorded in DataFrame
    period=0.1,
)
```
### Documentation

* New tutorial on `SystemSurrogate` (MR [#15](https://gitlab.com/cosapp/cosapp/-/merge_requests/15)).

### Bug fixes, minor improvements and code quality

* Initialization bug in time simulations (MR [#23](https://gitlab.com/cosapp/cosapp/-/merge_requests/23)).
* Bug in nonlinearity estimation in `NumericalSolver` (MR [#22](https://gitlab.com/cosapp/cosapp/-/merge_requests/22)).
* Do not raise `ArithmeticError` when an unknown is declared several time (MR [#18](https://gitlab.com/cosapp/cosapp/-/merge_requests/18)).
* Suppress deprecation warnings raised by `numpy` (MR [#20](https://gitlab.com/cosapp/cosapp/-/merge_requests/20) and [#24](https://gitlab.com/cosapp/cosapp/-/merge_requests/24)).
* Suppress undue warning raised by `numpy` in `NonLinearSolver` (MR [#19](https://gitlab.com/cosapp/cosapp/-/merge_requests/19)).
* Fix incompatibility between `pandas` and `xlrd` (MR [#21](https://gitlab.com/cosapp/cosapp/-/merge_requests/21)).
* Other code quality improvement (MR [#16](https://gitlab.com/cosapp/cosapp/-/merge_requests/16), [#17](https://gitlab.com/cosapp/cosapp/-/merge_requests/17), [#26](https://gitlab.com/cosapp/cosapp/-/merge_requests/26), [#27](https://gitlab.com/cosapp/cosapp/-/merge_requests/27)).

## 0.11.3 (2020-12-16)

### New features

* Surrogate models:
  It is now possible to create a surrogate model at any system level with new method `System.make_surrogate` (MR [#3](https://gitlab.com/cosapp/cosapp/-/merge_requests/3) and [#12](https://gitlab.com/cosapp/cosapp/-/merge_requests/12)):
  
```python
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
```

### Bug fixes, minor improvements and code quality

* Add several US-common unit conversions (MR [#2](https://gitlab.com/cosapp/cosapp/-/merge_requests/2)).
* New method to export cosapp system structure into a dictionary (MR [#5](https://gitlab.com/cosapp/cosapp/-/merge_requests/5))
* Make recorders capture port and system properties (MR [#8](https://gitlab.com/cosapp/cosapp/-/merge_requests/8)).
* Fix Module/System naming bug: *'inwards' and 'outwards' are allowed as Module/System names* (MR [#9](https://gitlab.com/cosapp/cosapp/-/merge_requests/9)).
* Broad code quality improvement (MR [#11](https://gitlab.com/cosapp/cosapp/-/merge_requests/11)).

  * Replace `typing.NoReturn` by `None` when appropriate.
  * Rewording pass, typo and error fixes in tutorial notebooks.
  * Suppress a `DeprecationWarning` raised by `numpy` in class `Variable`.
  * Reformat many strings as Python f-strings, for clarity.
  * Symplify many occurrences of `str.join()` for just two elements.

Global rewording of tutorial notebooks, including a few error fixes.

## 0.11.2 (2020-09-28)

First open-source version.
No major code change; mostly updates of license files, URLs in docs, and CI scripts.

## 0.11.1 (2020-07-22)

### Features

* Add the possibility to set boundary condition of transient simulation from interpolate profile.
* Add the possibility to prescribe a maximum step for transient variables.

### Bugs and code quality

* Bug fix in `RunOnce` driver, preventing undue call to `run_once` method.
* Bug fix in AssignString: force copy for numpy arrays.
* New tutorial for advanced features of time simulations in CosApp.

## 0.11.0 (2020-05-12)

### Features

* Improve documentation at various places, add documentation about the cosapp packages structure and sequence diagram for transient simulation.
* Add a advanced logger feature for CoSApp simulations.
* Update FMU export to PythonFMU 0.6.0
* New method `System.add_property` allowing users to create read-only properties.

### Bugs and code quality

* Suppress deprecation warnings raised by external dependencies. 
* Fix bug in AssignString with arrays, `AssignString` of the kind `'x = [0, 1, 2]'` won't change variable `x` into an array of integers,  if `x` is declared as an array of floats.
* Fix `TimeStackUnknown` not able to stack transient variables defined on a children System or with partially pulled transient variable.
* Fix the bug related to the initialization of `rate` attributes in systems.

## 0.10.2 (2020-04-21)

### Features

* [BETA] Export CoSApp System as FMU

### Bugs and code quality

* Apply Broyden correction on Jacobian matrix for iteration without Jacobian update
* Support varying time step
* Fix time not being set before `setup_run` are called.
* Fix reference for residues in `IterativeConnector` (it equals 1. now)
* Drop pyhamcrest for pytest

## 0.10.1 (2020-01-15)

### Features

* Time varying boundary conditions are now possible:

```python
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
```

### Bugs and code quality

* Fix various bug on the transient simulation front
* Correct implementation of step limitation in the Newton-Raphson solver
* Using a logger at `DEBUG` level will now display the call stack through the systems and drivers
* Rework of the Python evaluable string to be more efficient

## 0.10.0 (2019-10-23)

* Introduce continuous time simulations with dedicated time drivers (see `TimeDriver` notebook in tutorials).
* Suppress notion of (un)freeze; all variables are considered as known, unless explicitly declared as unknowns.
* Drivers no longer use ports.
* Connectors are now stored by parent system.
* Migrate to pytest.

### API Changes

* Ports:

  * `add_variable("x", units="m", types=Number)` => `add_variable("x", unit="m", dtype=Number)`
  * `freeze` => removed
  * `unfreeze` => replaced by `add_unknown` in Systems and Drivers
  * `connect_to` => replaced by `connect` at system level
  
* Systems:

  * `time_ref` is no longer an argument of method `compute`:
  
    `def compute(self, time_ref):` => `def compute(self):`
       
  * Create a new connection between `a.in1` and `b.out`:
  
    `self.a.in1.connect_to(self.b.out)` => `self.connect(self.a.in1, self.b.out)`
       
  * `add_residues` => `add_equation`
  * `set_numerical_default` => Pass keyword to `add_unknown`
  * `add_inward("x", units="m", types=Number)` => `add_inward("x", unit="m", dtype=Number)`
  * `add_outward("x", units="m", types=Number)` => `add_outward("x", unit="m", dtype=Number)`
  
* Drivers:

  * `add_unknowns(maximal_absolute_step, maximal_relative_step, low_bound, high_bound)` => `add_unknown(max_abs_step, max_rel_step, lower_bound, upper_bound)`
  * `add_equations` => `add_equation`
  * Equations are now represented by a unique string, instead of two strings (left-hand-side, right-hand-side):
  
    `add_equations("a", "b")` => `add_equation("a == b")`  
    
    `add_equations([("x", "2 * y + 1"), ("a", "b")])` => `add_equation(["x == 2 * y + 1", "a == b"])`  
        
  * For `NonLinearSolver`:
  
    `fatol` and `xtol` => `tol`  
    
    `maxiter` => `max_iter`  
        
  * For `Optimizer`:
  
    `ftol` => `tol`
    
    `maxiter` => `max_iter`

## 0.9.6 (2019-10-10)

* More correction for VISjs viewer and System HTML representation

## 0.9.5 (2019-09-25)

* Correct D3 & VISjs Viewers

## 0.9.4 (2019-09-25)

* Introduce an optional environment variable `COSAPP_CONFIG_DIR`

## 0.9.3 (2019-07-25)

### API Changes

* MonteCarlo:

  * `Montecarlo` => `MonteCarlo`
  * `Montecarlo.add_input_vars` => `MonteCarlo.add_random_variable`
  * `Montecarlo.add_response_vars` => `MonteCarlo.add_response`

* MonteCarlo has been improved by using Sobol random generator
* Viewers code on `System` is moved in a subpackage of `cosapp.tools`
* Residue reference is now calculated only once
* Various bug fix

## 0.9.2 (2019-07-01)
* In nonlinear solver, store LU factorization of the Jacobian matrix, rather than its inverse.
* Minor refactoring of the core source code, with no API changes

## 0.9.1 (2019-04-23)

* Create `Variable` class to manage variable attributes
* `watchdog` is now optional
* Configuration is now inside a folder `$HOME/.cosapp.d`
* API changes:
  - `get_latest_solution` => `save_solution`
  - `load_solver_solution` => `load_solution`
* Various bug fix

## 0.9.0 (2019-03-04)

This release introduces lots of API changes:

* Core ports and unit are available in `cosapp.ports`
* Core systems are available in `cosapp.systems`
* Core drivers are available in `cosapp.drivers`
* Core recorders are available in `cosapp.recorders`
* Core tools are available in `cosapp.tools`
* Core notebook tools are available in `cosapp.notebook` (! this is now a separated package)
* `data` have been renamed in `inwards` and `add_data` in `add_inward`
* `locals` have been renamed in `outwards` and `add_locals` in `add_outward`
* `BaseRecorder.record_iteration` renamed in `BaseRecorder.record_state`

- Huge code refractoring: cosapp is now a [Python namespace](https://packaging.python.org/guides/packaging-namespace-packages).
- `cosapp.notebook` has been moved to an independent package `cosapp_notebook`. But it is still accessible from `cosapp.notebook`.
- Introduce *Signal* / *Slot* pattern to connect to internal event (implementation from [signalslot](https://github.com/Numergy/signalslot), included in `cosapp.core.signal`)
    * `Module.setup_ran`: Signal emitted after the `call_setup_run` execution
    * `Module.computed`: Signal emitted after the full `compute` stack (i.e.: `_postcompute`)
    * `Module.clean_ran`: Signal emitted after the `call_clean_run` execution
    * `BaseRecorder.state_recorded`: Signale emitted after the `record_state` execution

## 0.8.0 (2018-10-26)

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

## 0.7.0 (2018-09-17)

- Add helper functions to present solver evolutions
- Add new d3 visualization of systems

## 0.6.0 (2018-08-14)

- Implement clean-dirty policy
- Restore compatibility with Python 3.4
- Display influence matrix

## 0.5.0 (2018-07-20)

- Simplify drivers structure, all actions for a case are supported by a single class `RunSingleCase`
- Add support for vector variables; they can be partially (un)frozen and are handled correctly by the solver.
- Add `MonteCarlo` driver
- Add recording data capability

## 0.4.0 (2018-06-15)

- `System` and `Driver` have now a common ancestor `Module` => `Driver` variables are now stored as data or locals
- Add visualization of `System` connections based on N2 graph (syntax: `cosapp.viewmodel(mySystem)`)

## 0.3.0 (2018-04-05)

**API changes:** `System.add_driver` and `Driver.add_child` take now an instance of `Driver`

- Add external code caller System
- Add validation range attributes on variables
- Add variable visibility
- Add metamodel training and DoE generator
- Add helper function to list inputs and outputs variables of a `System`

## 0.2.0 (2018-03-01)

* Stabilization of the user API

## 0.1.0 (2018-01-02)

* First release.
