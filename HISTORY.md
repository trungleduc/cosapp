# History


## 1.0.1 (2025-03-24)

### Bug fixes & Improvements

- New transient variables dynamically added during transitions are now accounted for in time drivers (MRs [#394](https://gitlab.com/cosapp/cosapp/-/merge_requests/394), [#396](https://gitlab.com/cosapp/cosapp/-/merge_requests/396)).
- Bug fixes in `CrankNicolson` (MRs [#395](https://gitlab.com/cosapp/cosapp/-/merge_requests/395), [#397](https://gitlab.com/cosapp/cosapp/-/merge_requests/397)).
- Refactoring of `state_io` functions (MR [#398](https://gitlab.com/cosapp/cosapp/-/merge_requests/398)).


## 1.0.0 (2025-03-12)

### New feature: Parallel execution of drivers

The parallelization of certain drivers was implemented by [Adrien Delsalle](https://gitlab.com/adriendelsalle) and [Gaétan Laurens](https://gitlab.com/GtnLrs) (MRs [#330](https://gitlab.com/cosapp/cosapp/-/merge_requests/330), [#332](https://gitlab.com/cosapp/cosapp/-/merge_requests/332), [#334](https://gitlab.com/cosapp/cosapp/-/merge_requests/334), [#335](https://gitlab.com/cosapp/cosapp/-/merge_requests/335), [#339](https://gitlab.com/cosapp/cosapp/-/merge_requests/339), [#341](https://gitlab.com/cosapp/cosapp/-/merge_requests/341), [#345](https://gitlab.com/cosapp/cosapp/-/merge_requests/345), [#348](https://gitlab.com/cosapp/cosapp/-/merge_requests/348), [#353](https://gitlab.com/cosapp/cosapp/-/merge_requests/353), [#357](https://gitlab.com/cosapp/cosapp/-/merge_requests/357), [#365](https://gitlab.com/cosapp/cosapp/-/merge_requests/365), [#374](https://gitlab.com/cosapp/cosapp/-/merge_requests/374), [#390](https://gitlab.com/cosapp/cosapp/-/merge_requests/390)):
- `LinearDoE` and `MonteCarlo`;
- `NonLinearSolver` (computation of the Jacobian matrix in parallel).

This major feature relies on a robust serialization of systems, drivers and recorders, using either `pickle` or `json`.
The serialization of all drivers, in particular, allows for the parallel execution of the drivers listed above, even when they contain sub-drivers.

#### Example

In the example below, using the `Ballistics` system from the advanced time driver tutorial, we compute the initial velocity leading to a targetted end point, after an imposed flight time of two seconds.
This design case implies a `RungeKutta` time driver, embedded into a `NonLinearSolver` driver.
For the latter, we request a parallel computation of the Jacobian matrix by forward finite differences, using three workers.

```python
from cosapp.drivers import NonLinearSolver, RungeKutta
from cosapp.core.numerics.solve import FfdJacobianEvaluation
from cosapp.utils.execution import ExecutionPolicy, ExecutionType

# Set test case
point = Ballistics("point")
ncpus = 3  # compute the Jacobian matrix using 3 workers

solver = point.add_driver(
    NonLinearSolver(
        "solver",
        jac=FfdJacobianEvaluation(
            execution_policy=ExecutionPolicy(ncpus, ExecutionType.MULTI_PROCESSING)
        ),
    )
)
driver = solver.add_child(RungeKutta(time_interval=(0, 2), dt=0.1, order=3))

# Set design problem: compute initial velocity that leads to a target point
solver.add_unknown("v0").add_equation("x == [10, 0, 0]")

# Define a time simulation scenario
driver.set_scenario(
    init={"x": [0, 0, 0], "v": "v0"},
    values={"point.mass": 1.5, "point.k": 0.9},
)

# Set initial guess & solve
point.v0 = np.ones(3)
point.run_drivers()
```

### Other new features & API changes

- Implementation of second-order, implicit time driver `CrankNicolson` (MRs [#377](https://gitlab.com/cosapp/cosapp/-/merge_requests/377)).
  This driver essentially works like a nonlinear solver, solving for both transient variables and intrinsic unknowns of the system of interest.
  Therefore, it is highly recommended for dynamic systems containing algebraic loops or intrinsic constraints, instead of nesting a nonlinear solver inside an explicit time driver.
- Detect the occurrence of multiple primary events at the same time (MR [#376](https://gitlab.com/cosapp/cosapp/-/merge_requests/376)).
- Add the possibility to filter events with a Boolean expression evaluated in a specific context, other than that of the event owner (MR [#375](https://gitlab.com/cosapp/cosapp/-/merge_requests/375)).
- Add `numpy.where` and `logspace` to the scope of evaluable expressions, used in equations and boundary conditions, *e.g.* (MR [#382](https://gitlab.com/cosapp/cosapp/-/merge_requests/382)).

### Bug fixes and code improvements

- Bug fix in `EvalString` serialization (MR [#383](https://gitlab.com/cosapp/cosapp/-/merge_requests/383)).
- Bug fix in the serialization of events (MR [#386](https://gitlab.com/cosapp/cosapp/-/merge_requests/386)).
- Bug fix in `MonteCarlo` with a time subdriver (MRs [#378](https://gitlab.com/cosapp/cosapp/-/merge_requests/378), [#392](https://gitlab.com/cosapp/cosapp/-/merge_requests/392)).
- Prevent name clashes for ports and sub-systems (MR [#381](https://gitlab.com/cosapp/cosapp/-/merge_requests/381)).
- Other bug fixes (MRs [#379](https://gitlab.com/cosapp/cosapp/-/merge_requests/379), [#380](https://gitlab.com/cosapp/cosapp/-/merge_requests/380), [#389](https://gitlab.com/cosapp/cosapp/-/merge_requests/389)).

### Documentation

- Add Crank-Nicolson driver in time driver tutorial (MR [#385](https://gitlab.com/cosapp/cosapp/-/merge_requests/385)).
- New tutorial on multiprocessing (MR [#391](https://gitlab.com/cosapp/cosapp/-/merge_requests/391)).

### Dependency management

- Compatibility with `numpy` 2, with retrocompatibility with version 1 (MRs [#309](https://gitlab.com/cosapp/cosapp/-/merge_requests/309), [#387](https://gitlab.com/cosapp/cosapp/-/merge_requests/387)).
- Compatibility with `pytest` 8.3 was restored (MR [#388](https://gitlab.com/cosapp/cosapp/-/merge_requests/388)).
- The code is no longer compatible with Python 3.8, as it now contains type hints of the kind `list[str]` or `dict[str, float]` (instead of `typing.List` and `typing.Dict`), introduced in Python 3.9.


## 0.19.2 (2025-01-23)

### Bug fixes and code improvements

- Improve the system transition mechanism (MR [#370](https://gitlab.com/cosapp/cosapp/-/merge_requests/370)).
- Fix bug in event cascade resolution (MRs [#371](https://gitlab.com/cosapp/cosapp/-/merge_requests/371) & [#372](https://gitlab.com/cosapp/cosapp/-/merge_requests/372)).

### Maintenance and code quality

- Improve type hints (MR [#369](https://gitlab.com/cosapp/cosapp/-/merge_requests/369)).


## 0.19.1 (2025-01-07)

### Bug fixes and code quality

- Fix dealiasing bug for custom objects (MR [#362](https://gitlab.com/cosapp/cosapp/-/merge_requests/362)).
- Fix bug in `ExplicitTimeDriver.set_scenario` (MR [#363](https://gitlab.com/cosapp/cosapp/-/merge_requests/363)).
- Fix bug with array transients (MR [#364](https://gitlab.com/cosapp/cosapp/-/merge_requests/364)).
- Suppress unnecessary exception in `PeriodicEvent.tick` (MR [#366](https://gitlab.com/cosapp/cosapp/-/merge_requests/366)).
- Improve the handling of timed events (MR [#367](https://gitlab.com/cosapp/cosapp/-/merge_requests/367)).


## 0.19.0 (2024-12-04)

### New features & API changes

- Improve the performance of unknowns and residues (MRs [#350](https://gitlab.com/cosapp/cosapp/-/merge_requests/350), [#352](https://gitlab.com/cosapp/cosapp/-/merge_requests/352), by [Gaétan Laurens](https://gitlab.com/GtnLrs)).
  In particular, this version introduces the possiblity to declare as unknown any settable attribute of an input object (that is an object contained in an input port), which until now was only possible for input port variables.
  Example:

  ```python
  from cosapp.base import System
  from cosapp.drivers import NonLinearSolver
  from dataclasses import dataclass

  @dataclass
  class Foo:
    a: float
    x: float

  class SomeSystem(System):
      def setup(self):
          self.add_inward("foo", Foo(a=1.0, x=1.0))
          self.add_outward("y", 0.0)

      def compute(self):
          self.y = self.foo.a - self.foo.x**3
  
  system = SomeSystem("system")
  solver = system.add_driver(NonLinearSolver("solver"))

  solver.add_unknown("foo.x")   # attribute `x` can now be manipulated by the solver
  solver.add_equation("y == 0")

  system.foo.a = -2.0
  system.run_drivers()

  import pytest, math
  assert system.foo.x == pytest.approx(math.cbrt(system.foo.a))
  ```

- Fix bug in `TwoPointCubicInterpolator` with multidimensional arrays (MR [#355](https://gitlab.com/cosapp/cosapp/-/merge_requests/355)).
- Allow filtered events in merged events (MR [#356](https://gitlab.com/cosapp/cosapp/-/merge_requests/356)):

  ```python
  system.event.trigger = Event.merge(
      event_a,
      event_b.filter("x > 0"),
  )
  ```

- Introduction of periodic events (MR [#360](https://gitlab.com/cosapp/cosapp/-/merge_requests/360)):

  ```python
  from cosapp.multimode import PeriodicTrigger

  system.event_a.trigger = PeriodicTrigger(period=2.3)
  system.event_b.trigger = PeriodicTrigger(period=0.25, t0=0.33)
  ```

### Documentation

- Add instructions on how to perform benchmarks with `asv_runner` in main README file (MR [#358](https://gitlab.com/cosapp/cosapp/-/merge_requests/358)).

### Maintenance and code quality

- Fix import error in JupyterLite (MR [#351](https://gitlab.com/cosapp/cosapp/-/merge_requests/351)).
- Minor code refactoring (MR [#354](https://gitlab.com/cosapp/cosapp/-/merge_requests/354)).
- Use public runners for all CI jobs (MR [#359](https://gitlab.com/cosapp/cosapp/-/merge_requests/359)).


## 0.18.0 (2024-10-24)

### New features & API changes

- Improved data recording mechanism for `NonLinearSolver`, to facilitate convergence path analysis (MR [#312](https://gitlab.com/cosapp/cosapp/-/merge_requests/312)).
- New driver `FixedPointSolver`, solving algebraic loops by fixed-point iterations (MRs [#315](https://gitlab.com/cosapp/cosapp/-/merge_requests/315) & [#322](https://gitlab.com/cosapp/cosapp/-/merge_requests/322)).
- New method `Driver.available_options`, returning the list of options available for a particular driver (MR [#316](https://gitlab.com/cosapp/cosapp/-/merge_requests/316)).
- New method `System.pull_design_method`, to promote sub-system design methods at parent level easily (MR [#314](https://gitlab.com/cosapp/cosapp/-/merge_requests/314)).
- Bug fix in FMU exporter (MR [#324](https://gitlab.com/cosapp/cosapp/-/merge_requests/324)).
- Bug fixes in system transition logic (MR [#338](https://gitlab.com/cosapp/cosapp/-/merge_requests/338)).
- New method `System.init_mode` called before each time simulation, for mode initialization (MR [#347](https://gitlab.com/cosapp/cosapp/-/merge_requests/347)).

### Documentation

- Fix bad rendering of tutorial notebooks on time simulations (MR [#321](https://gitlab.com/cosapp/cosapp/-/merge_requests/321)).
- Fix bad rendering of mermaid graph in logger documentation (MR [#326](https://gitlab.com/cosapp/cosapp/-/merge_requests/326)).
- Updated tutorial on advanced time simulations and on design methods (MR [#343](https://gitlab.com/cosapp/cosapp/-/merge_requests/343)).
- Add a section on `FixedPointSolver` in the driver tutorial (MR [#346](https://gitlab.com/cosapp/cosapp/-/merge_requests/346)).

### Maintenance and code quality

- Improved type hints for drivers (MR [#318](https://gitlab.com/cosapp/cosapp/-/merge_requests/318)).
- Updated JupyterLite image (MR [#313](https://gitlab.com/cosapp/cosapp/-/merge_requests/313)).
- Add new test on `swap_system` (MR [#342](https://gitlab.com/cosapp/cosapp/-/merge_requests/342)).
- Various updates of CI/CD settings (MRs [#317](https://gitlab.com/cosapp/cosapp/-/merge_requests/317), [#319](https://gitlab.com/cosapp/cosapp/-/merge_requests/319) [#320](https://gitlab.com/cosapp/cosapp/-/merge_requests/320), [#336](https://gitlab.com/cosapp/cosapp/-/merge_requests/336), [#337](https://gitlab.com/cosapp/cosapp/-/merge_requests/337)).
- Various test improvements (MRs [#340](https://gitlab.com/cosapp/cosapp/-/merge_requests/340), [#342](https://gitlab.com/cosapp/cosapp/-/merge_requests/342)).


## 0.17.0 (2024-06-18)

### New features & API changes

- A JupyterLite image including CoSApp is now available in the main README file (MR [#300](https://gitlab.com/cosapp/cosapp/-/merge_requests/300)).
- Fix a bug with primary event initialization (MR [#293](https://gitlab.com/cosapp/cosapp/-/merge_requests/293)).
- Improved API for partial connections: name mappings can now be given as lists mixing variable names and dictionaries, which is convenient when most variable names are identical, and only a few differ (MRs [#294](https://gitlab.com/cosapp/cosapp/-/merge_requests/294), [#295](https://gitlab.com/cosapp/cosapp/-/merge_requests/295) & [#306](https://gitlab.com/cosapp/cosapp/-/merge_requests/306)).
  Example:

  ```python
  from cosapp.base import System
  
  class SomeSystem(System):
      def setup(self):
          foo = self.add_child(Foo('foo'), pulling=['a', 'b', {'c': 'c_foo'}])
          bar = self.add_child(Bar('bar'))

          self.connect(foo, bar, ['x', 'y', {'z': 'v'}])
  ```
  
- New utility function `cosapp.tools.views.show_tree` displaying the hierarchical tree of a system similar to a folder tree in a filesystem (MRs [#296](https://gitlab.com/cosapp/cosapp/-/merge_requests/296)-[#298](https://gitlab.com/cosapp/cosapp/-/merge_requests/298)).

### Bug fixes and code quality

- New attribute `Residue.variables` providing the names of the variables involved in the residue (MR [#299](https://gitlab.com/cosapp/cosapp/-/merge_requests/299)).
- Minor bug fix in Newton-Raphson algorithm (MR [#303](https://gitlab.com/cosapp/cosapp/-/merge_requests/303)).
- Fix bad markdown rendering of systems (MR [#292](https://gitlab.com/cosapp/cosapp/-/merge_requests/292)).
- Improved type hints throughout the code (MR [#307](https://gitlab.com/cosapp/cosapp/-/merge_requests/307)).
- Other improvements (MRs [#291](https://gitlab.com/cosapp/cosapp/-/merge_requests/291), [#301](https://gitlab.com/cosapp/cosapp/-/merge_requests/301), [#302](https://gitlab.com/cosapp/cosapp/-/merge_requests/302), [#304](https://gitlab.com/cosapp/cosapp/-/merge_requests/304), [#305](https://gitlab.com/cosapp/cosapp/-/merge_requests/305), [#307](https://gitlab.com/cosapp/cosapp/-/merge_requests/307), [#308](https://gitlab.com/cosapp/cosapp/-/merge_requests/308)).

### Maintenance

- Pin dependency to `numpy` v1, until full migration to v2 (MR [#310](https://gitlab.com/cosapp/cosapp/-/merge_requests/310)).


## 0.16.0 (2024-04-18)

### Bug fixes and code quality

- Primary events triggered within the same time step are now correctly captured (MR [#287](https://gitlab.com/cosapp/cosapp/-/merge_requests/287)).
  A bug persists when several primary events occur at the exact same time, though, as only one will be retained.
- Fix bug preventing events from occurring during the first time step of a simulation (MR [#288](https://gitlab.com/cosapp/cosapp/-/merge_requests/288)).
- Fix a bug causing recorders to crash when inspecting ports with properties (MR [#282](https://gitlab.com/cosapp/cosapp/-/merge_requests/282)).
- Algebraic and time-dependent problems are now dissociated (MR [#284](https://gitlab.com/cosapp/cosapp/-/merge_requests/284)). As a consequence, invoking `self.problem.clear()` during transitions, for instance, no longer affects time-dependent unknowns such as transients.
- Fix ambiguous warning message raised by `RunSingleCase` (MR [#279](https://gitlab.com/cosapp/cosapp/-/merge_requests/279)).
- Various refactoring passes (MRs [#283](https://gitlab.com/cosapp/cosapp/-/merge_requests/283) and [#286](https://gitlab.com/cosapp/cosapp/-/merge_requests/286)).

### Maintenance

- Updated description inside the conda recipe (MR [#276](https://gitlab.com/cosapp/cosapp/-/merge_requests/276)).
- Pin `pytest` version due to a bug in version 8.1 (MR [#280](https://gitlab.com/cosapp/cosapp/-/merge_requests/280)).
- Add a "Citing" section in the main README file (MR [#281](https://gitlab.com/cosapp/cosapp/-/merge_requests/281)).
- The module parser was updated (MR [#285](https://gitlab.com/cosapp/cosapp/-/merge_requests/285)).
- Force `sphinx` < 7.3 in the documentation building environment, owing to an incompatibility with `sphinx-mdinclude` (MR [#289](https://gitlab.com/cosapp/cosapp/-/merge_requests/289)). This is a temporary patch until root cause is fixed.


## 0.15.4 (2024-02-28)

### Bug fixes and code quality

- Fix bug raised in `NonLinearSolver` for systems with rates (MR [#268](https://gitlab.com/cosapp/cosapp/-/merge_requests/268)).
- Minor refactoring pass (MR [#274](https://gitlab.com/cosapp/cosapp/-/merge_requests/274)).

### Maintenance

- Drop Python 3.8 support and add Python 3.11 test pipeline (MR [#270](https://gitlab.com/cosapp/cosapp/-/merge_requests/270)).
- Update installation and contribution guidelines (MR [#270](https://gitlab.com/cosapp/cosapp/-/merge_requests/270)).
- Update dependency list (MRs [#269](https://gitlab.com/cosapp/cosapp/-/merge_requests/269) and [#270](https://gitlab.com/cosapp/cosapp/-/merge_requests/270)).

### JOSS article

Publication of an article on CoSApp in the [Journal of Open-Source Software](https://joss.theoj.org/), referenced to version 0.15.4 (MR [#271](https://gitlab.com/cosapp/cosapp/-/merge_requests/271)).


## 0.15.3 (2023-12-19)

### Bug fixes and code quality

- Fix minor bug in `NonLinearSolver` (MR [#261](https://gitlab.com/cosapp/cosapp/-/merge_requests/261)).
- Fix incorrect return type hints in `System` (MR [#259](https://gitlab.com/cosapp/cosapp/-/merge_requests/259)).
- Improve support for read-only properties in system getters and setters (MR [#264](https://gitlab.com/cosapp/cosapp/-/merge_requests/264)).

### Maintenance

- Remove deprecated dependency for compatibility with Python 3.12 (MR [#258](https://gitlab.com/cosapp/cosapp/-/merge_requests/258)).
- Force `pythonfmu` < 0.6.3 in dependency list, to prevent a crash during tests (MR [#262](https://gitlab.com/cosapp/cosapp/-/merge_requests/262)). Temporary fix until root cause is identified.


## 0.15.2 (2023-10-14)

### Bug fixes

- Fix crash with `VisJs` rendering of sub-systems (MR [#252](https://gitlab.com/cosapp/cosapp/-/merge_requests/252)).
- Fix minor bug in `EvalString` (MR [#253](https://gitlab.com/cosapp/cosapp/-/merge_requests/253)).
- Fix bad synchronization of mode variables during transitions (MR [#254](https://gitlab.com/cosapp/cosapp/-/merge_requests/254)).
- Improve error message when pulling variables with different roles (MR [#255](https://gitlab.com/cosapp/cosapp/-/merge_requests/255)).
- Fix `TypeError` when passing `pulling` argument as a tuple in `System.add_child` (MR [#256](https://gitlab.com/cosapp/cosapp/-/merge_requests/256)).


## 0.15.1 (2023-09-18)

### New features & API changes

- The representation of `MathematicalProblem` objects now indicates the number of unknowns and equations, for a better readability (MR [#249](https://gitlab.com/cosapp/cosapp/-/merge_requests/249)).
- Optional argument `execution_index` in `System.add_child` and `Driver.add_child` can now take a negative value, with a behaviour following that of `list.insert` (MR [#250](https://gitlab.com/cosapp/cosapp/-/merge_requests/250)).

### Bug fixes and code quality

- Add missing field "time" in `ExplicitTimeRecorder.event_data`, when no recorder is set (MR [#245](https://gitlab.com/cosapp/cosapp/-/merge_requests/245)).
- Fix bug with recorders inspecting systems with iterators (MRs [#246](https://gitlab.com/cosapp/cosapp/-/merge_requests/246) and [#247](https://gitlab.com/cosapp/cosapp/-/merge_requests/247)).
- Fix bug with `NonLinearSolver` with NumPy array residues (MR [#248](https://gitlab.com/cosapp/cosapp/-/merge_requests/248)).
- Fix initialization bug of targets involving expressions (MR [#249](https://gitlab.com/cosapp/cosapp/-/merge_requests/249)).


## 0.15.0 (2023-07-20)

### New features & API changes

- Suppression of default `RunSingleCase` subdriver `runner` in `NonLinearSolver` drivers (MR [#239](https://gitlab.com/cosapp/cosapp/-/merge_requests/239)).
- Enable target initialization in multi-point design problems (MR [#233](https://gitlab.com/cosapp/cosapp/-/merge_requests/233)).
- New utility function `swap_system` to replace on the fly a subsystem by another `System` instance (MR [#238](https://gitlab.com/cosapp/cosapp/-/merge_requests/238)).

### Bug fixes and code quality

- Refactor driver `Optimizer` (MR [#240](https://gitlab.com/cosapp/cosapp/-/merge_requests/240)).
- Various bug fixes (MRs [#234](https://gitlab.com/cosapp/cosapp/-/merge_requests/234), [#235](https://gitlab.com/cosapp/cosapp/-/merge_requests/235), [#241](https://gitlab.com/cosapp/cosapp/-/merge_requests/241)).

### Documentation

- New tutorial on `swap_system` (MR [#243](https://gitlab.com/cosapp/cosapp/-/merge_requests/243)).
- General update of tutorials (MRs [#232](https://gitlab.com/cosapp/cosapp/-/merge_requests/232) and [#242](https://gitlab.com/cosapp/cosapp/-/merge_requests/242)).


## 0.14.1 (2023-06-08)

### Bug fixes and code quality

- Fix incorrect output file name in `cosapp.tools.parse_module` (MR [#229](https://gitlab.com/cosapp/cosapp/-/merge_requests/229)).
- Refactoring pass (MR [#230](https://gitlab.com/cosapp/cosapp/-/merge_requests/230)).

### Documentation

- Upgrade documentation build stack (MRs [#224](https://gitlab.com/cosapp/cosapp/-/merge_requests/224) and [#226](https://gitlab.com/cosapp/cosapp/-/merge_requests/226)).
- Fix bad rendering of ports and systems (MRs [#227](https://gitlab.com/cosapp/cosapp/-/merge_requests/227) and [#228](https://gitlab.com/cosapp/cosapp/-/merge_requests/228)).


## 0.14.0 (2023-05-17)

### New features & API changes

* Improved performance, through a revised clean/dirty mechanism (MR [#215](https://gitlab.com/cosapp/cosapp/-/merge_requests/215)).
* Possibility to add a contextual description to sub-systems and ports of a system, as well as sub-drivers (MR [#216](https://gitlab.com/cosapp/cosapp/-/merge_requests/216)). This feature is useful for automatic documentation tools, and has been included in the Markdown representation of systems (also used in function `cosapp.tools.display_doc`). Example:
  
  ```python
  from cosapp.base import System
  from my_module import FlowPort
  
  class MySystem(System):
      def setup(self):
          self.add_input(FlowPort, "fl_in1", desc="Primary inlet flow port")
          self.add_input(FlowPort, "fl_in2", desc="Secondary inlet flow port")
  
          self.add_output(FlowPort, "fl_out")
  ```
  
* New hook function `_parse_module_config`, returning pre-defined settings for `cosapp.tools.parse_module` (MR [#218](https://gitlab.com/cosapp/cosapp/-/merge_requests/218)). This allows module maintainers to simply call
  
  ```python
  from cosapp.tools import parse_module
  import my_module
  
  parse_module(my_module)
  ```

  instead of, *e.g.*,

  ```python
  parse_module(
      my_module,
      ctor_config={
          "ComplexSystem1": [
              dict(n=1, foo=0.5),
              dict(n=2, foo=0.1),
          ],
          "ComplexSystem2": [
              dict(xi=0.0, __alias__="ComplexSystem2_a"),
              dict(xi=1.0, __alias__="ComplexSystem2_b"),
          ],
      },
      excludes=["Foo*", "*Bar?"],
  )
  ```

  provided `my_module._parse_module_config()` returns a dictionary specifying the values of `ctor_config`, `excludes`, *etc.*

* Make `SolverResults` a `dataclass`, for easier handling of `NonLinearSolver.results`, *e.g.* (MR [#220](https://gitlab.com/cosapp/cosapp/-/merge_requests/220)).
* Expose attribute `problem` in system setup (MR [#221](https://gitlab.com/cosapp/cosapp/-/merge_requests/221)). Previously, `problem` was only exposed in method `System.transition`.

### Bug fixes and code quality

* Fix bug in `NonLinearSolver` log message (MR [#214](https://gitlab.com/cosapp/cosapp/-/merge_requests/214)).
* Add missing JSON file in PyPI and conda packages, preventing the use of `cosapp.tools.parse_module` (MR [#219](https://gitlab.com/cosapp/cosapp/-/merge_requests/219)).

### Documentation

* Illustrate port and sub-system description in tutorials (MR [#215](https://gitlab.com/cosapp/cosapp/-/merge_requests/215)).


## 0.13.1 (2023-04-03)

### Bug fixes and code quality

* Update test baseline to pass in a Python 3.11 environment (MR [#200](https://gitlab.com/cosapp/cosapp/-/merge_requests/200)).
* Creation of a dummy system factory, mostly for tests (MR [#201](https://gitlab.com/cosapp/cosapp/-/merge_requests/201)).
* Improve clean/dirty mechanism (MR [#202](https://gitlab.com/cosapp/cosapp/-/merge_requests/202)) and fix incorrect clean/dirty status with surrogate models (MR [#205](https://gitlab.com/cosapp/cosapp/-/merge_requests/205)).
* Fix bug in function `cosapp.tools.display_doc` for classes with setup arguments (MR [#208](https://gitlab.com/cosapp/cosapp/-/merge_requests/208)).
* Various code quality improvements (MRs [#211](https://gitlab.com/cosapp/cosapp/-/merge_requests/211), [#212](https://gitlab.com/cosapp/cosapp/-/merge_requests/212)).

### Documentation

Note: extensive use of self-documenting f-strings (introduced in Python 3.8) has made tutorials incompatible with Python 3.7.

* Updated tutorials (MRs [#203](https://gitlab.com/cosapp/cosapp/-/merge_requests/203), [#206](https://gitlab.com/cosapp/cosapp/-/merge_requests/206), [#207](https://gitlab.com/cosapp/cosapp/-/merge_requests/207), [#210](https://gitlab.com/cosapp/cosapp/-/merge_requests/210)).
* The image used in binder is now based on Python 3.10 (MR [#204](https://gitlab.com/cosapp/cosapp/-/merge_requests/204)).

### Module parser

* New function `parse_module` in `cosapp.tools` collecting all system and port classes within a Python module. This parser, generating a JSON file containing the description of all CoSApp symbols, is primarily meant to be used for constructional GUI applications, developed separately (MRs [#192](https://gitlab.com/cosapp/cosapp/-/merge_requests/192) & [#209](https://gitlab.com/cosapp/cosapp/-/merge_requests/209)).


## 0.13.0 (2023-02-09)

### Python 3.10 support

The code is now tested for Python 3.8, 3.9 and 3.10.
Support of Python 3.7 is thus officially dropped, although no version-specific Python code was introduced in this version of CoSApp.

### New features & API changes

* Module `connectors` moved from `cosapp.core` to `cosapp.ports` (MR [#189](https://gitlab.com/cosapp/cosapp/-/merge_requests/189)).
* New "direct" (with no unit conversion) connector classes `PlainConnector`, `CopyConnector` and `DeepCopyConnector`, in `cosapp.ports.connectors` (MR [#188](https://gitlab.com/cosapp/cosapp/-/merge_requests/188)).
* New method `MathematicalProblem.is_empty()`, equivalent to `shape == (0, 0)` (MR [#186](https://gitlab.com/cosapp/cosapp/-/merge_requests/186)).
* Improved VisJs graph rendering, by limiting node size for long system names (MR [#184](https://gitlab.com/cosapp/cosapp/-/merge_requests/184)).
* New utility functions `get_state` and `set_state` in `cosapp.utils`, for quick system data recovery (MR [#193](https://gitlab.com/cosapp/cosapp/-/merge_requests/193)):

  ```python
  from cosapp.utils import get_state, set_state
  
  s = SomeSystem('s')
  # ... many design steps, say
  
  # Save state in local object
  designed = get_state(s)
  
  s.drivers.clear()
  s.add_driver(SomeDriver('driver'))
  
  try:
      s.run_drivers()
  except:
      # Recover previous state
      set_state(s, designed)
  ```

* Functions `radians`, `degrees` and `arctan2`/`atan2` have been added to the scope of `EvalString` objects, and can therefore be used in equations, *e.g.* (MR [#178](https://gitlab.com/cosapp/cosapp/-/merge_requests/178)).
* Recorders can now record constant properties (MR [#181](https://gitlab.com/cosapp/cosapp/-/merge_requests/181)).
* Deprecation of `System.get_unsolved_problem` in favour of new method `assembled_problem` (MR [#174](https://gitlab.com/cosapp/cosapp/-/merge_requests/174)).
* Inner off-design problem of systems is now exposed as attribute `problem`, but only within the `transition` method (MR [#174](https://gitlab.com/cosapp/cosapp/-/merge_requests/174)). This allows users to add or remove off-design constraints during event-driven transitions, while keeping this property inaccessible the rest of the time.

  ```python
  from cosapp.base import System
  from math import sin, cos
  
  class SomeSystem(System):
      def setup(self):
          self.add_inward('x', 0.0)
          self.add_inward('y', 0.0)
          self.add_outward('z', 0.0)
          a = self.add_event('event_a', trigger='x > y')
          b = self.add_event('event_b', trigger='x < y')
      
      def compute(self):
          self.z = cos(self.x) * sin(self.y)
  
      def transition(self):
          offdesign = self.problem
          if self.event_a.present:
              offdesign.clear()
              offdesign.add_equation('z == 0.5').add_unknown('x')
          if self.event_b.present:
              offdesign.clear()
  ```

### Bug fixes and code quality

* Fix serialization bugs for systems with setup parameters (MR [#180](https://gitlab.com/cosapp/cosapp/-/merge_requests/180)) and `None` variables (MR [#172](https://gitlab.com/cosapp/cosapp/-/merge_requests/172)).
* Bug fix on event time calculation involving array transients (MR [#177](https://gitlab.com/cosapp/cosapp/-/merge_requests/177)).
* Bug fix on possible name conflicts in connector storage (MR [#173](https://gitlab.com/cosapp/cosapp/-/merge_requests/173)).
* Fix inconsistent behaviour of `System.add_child` when a `pulling` error is raised (MR [#197](https://gitlab.com/cosapp/cosapp/-/merge_requests/197)).
* Various code quality improvements (MRs [#175](https://gitlab.com/cosapp/cosapp/-/merge_requests/175), [#185](https://gitlab.com/cosapp/cosapp/-/merge_requests/185), [#186](https://gitlab.com/cosapp/cosapp/-/merge_requests/186), [#187](https://gitlab.com/cosapp/cosapp/-/merge_requests/187), [#190](https://gitlab.com/cosapp/cosapp/-/merge_requests/190), [#191](https://gitlab.com/cosapp/cosapp/-/merge_requests/191), [#194](https://gitlab.com/cosapp/cosapp/-/merge_requests/194)).

### Documentation

* Updated time driver tutorial (MR [#182](https://gitlab.com/cosapp/cosapp/-/merge_requests/182)).
* Updated Tips & Tricks (MR [#196](https://gitlab.com/cosapp/cosapp/-/merge_requests/196)).
* Other updates (MRs [#176](https://gitlab.com/cosapp/cosapp/-/merge_requests/176), [#195](https://gitlab.com/cosapp/cosapp/-/merge_requests/195), [#198](https://gitlab.com/cosapp/cosapp/-/merge_requests/198)).

## 0.12.3 (2022-09-21)

### New features & API changes

* Improved user experience:
  * Auto-completion for dynamically added attributes of systems (ports, subsystems, inwards, outwards, events...) and drivers (sub-drivers) (MRs [#164](https://gitlab.com/cosapp/cosapp/-/merge_requests/164) and [#166](https://gitlab.com/cosapp/cosapp/-/merge_requests/166));
  * Improved representation of mathematical problems, in particular for loop equations and multi-point design problems (MRs [#159](https://gitlab.com/cosapp/cosapp/-/merge_requests/159), [#160](https://gitlab.com/cosapp/cosapp/-/merge_requests/160) and [#163](https://gitlab.com/cosapp/cosapp/-/merge_requests/163)).
* New method `System.new_problem` to facilitate the creation of dynamic design methods, *e.g.* (MR [#162](https://gitlab.com/cosapp/cosapp/-/merge_requests/162)).

### Documentation

* Updated tutorials (MRs [#169](https://gitlab.com/cosapp/cosapp/-/merge_requests/169) and [#170](https://gitlab.com/cosapp/cosapp/-/merge_requests/170)).

### Bug fixes and code quality

* Bug fix on unknown dealiasing (MR [#157](https://gitlab.com/cosapp/cosapp/-/merge_requests/157)).
* Bug fix on port variable export (MR [#158](https://gitlab.com/cosapp/cosapp/-/merge_requests/158)).
* Bug fix on system serialization (MR [#161](https://gitlab.com/cosapp/cosapp/-/merge_requests/161)).
* Bug fix on subsystem read-only constants (MR [#167](https://gitlab.com/cosapp/cosapp/-/merge_requests/167)).
* Various code quality improvements (MRs [#154](https://gitlab.com/cosapp/cosapp/-/merge_requests/154), [#155](https://gitlab.com/cosapp/cosapp/-/merge_requests/155), [#162](https://gitlab.com/cosapp/cosapp/-/merge_requests/162), [#168](https://gitlab.com/cosapp/cosapp/-/merge_requests/168)).

## 0.12.2 (2022-05-26)

### New features & API changes

* Automatic tolerance in `NonLinearSolver` (MRs [#148](https://gitlab.com/cosapp/cosapp/-/merge_requests/148) and [#152](https://gitlab.com/cosapp/cosapp/-/merge_requests/152)).
* `Optimizer.set_objective` is deprecated, in favour of `set_minimum` and `set_maximum` (MR [#150](https://gitlab.com/cosapp/cosapp/-/merge_requests/150)).
* Loop residues are no longer normalized (MR [#151](https://gitlab.com/cosapp/cosapp/-/merge_requests/151)).
* Improved type hints for ports, systems and drivers, in functions whose return type depends on arguments, such as `System.add_child`, `add_driver`, *etc.* (MR [#145](https://gitlab.com/cosapp/cosapp/-/merge_requests/145)).

### Documentation

* Updated tutorials (MRs [#144](https://gitlab.com/cosapp/cosapp/-/merge_requests/144), [#147](https://gitlab.com/cosapp/cosapp/-/merge_requests/147) and [#154](https://gitlab.com/cosapp/cosapp/-/merge_requests/154)).

### Bug fixes and code quality

* Fix inconsistent behaviour of recorders (MR [#143](https://gitlab.com/cosapp/cosapp/-/merge_requests/143)).
* Fix bug on event initialization (MR [#146](https://gitlab.com/cosapp/cosapp/-/merge_requests/146)).
* Fix bad markdown formatting of ports occurring in `jupyterlab` 3.4 (MR [#149](https://gitlab.com/cosapp/cosapp/-/merge_requests/149)).
* Fix failed test caused by a regression in `pytest` 7.1 (MR [#142](https://gitlab.com/cosapp/cosapp/-/merge_requests/142)).

## 0.12.1 (2022-02-25)

### New features & API changes

* Simplification of driver `Optimizer`:
  * Suppression of sub-driver `runner` (MR [#136](https://gitlab.com/cosapp/cosapp/-/merge_requests/136)). This change introduces new methods `set_objective`, `add_unknowns` and `add_constraints` in driver `Optimizer`.
  * Optimization constraints are now declared with human-readable expressions in `Optimizer.add_constraints` (MR [#138](https://gitlab.com/cosapp/cosapp/-/merge_requests/138)).

  Before:
    
  ```python
  from cosapp.drivers import Optimizer
  
  s = SomeSystem('s')
  optim = s.add_driver(Optimizer('optim'))
  
  optim.runner.set_objective('cost')
  optim.runner.add_unknown(['a', 'b', 'p_in.x'])
  # Enter constraints as non-negative expressions:
  optim.runner.add_constraints([
      "b - a",  # b >= a
      "a",      # a >= 0
      "1 - a",  # a <= 1
  ])
  optim.runner.add_constraints(
      "p_out.y",
      inequality = False,  # p_out.y == 0
  )
  
  s.run_drivers()
  ```
  
  After:
    
  ```python
  optim.set_objective('cost')
  optim.add_unknown(['a', 'b', 'p_in.x'])
  optim.add_constraints(
      "b >= a",
      "0 <= a <= 1",
      "p_out.y == 0",
  )
  ```

  * New, convenient iterators and setters for ports (MR [#137](https://gitlab.com/cosapp/cosapp/-/merge_requests/137)):
  
  ```python
  from cosapp.base import Port, System
  
  class XyzPort(Port):
      def setup(self):
          self.add_variable('x')
          self.add_variable('y')
          self.add_variable('z')
  
  class SomeSystem(System):
      def setup(self):
          self.add_input(XyzPort, 'p_in')
          self.add_output(XyzPort, 'p_out')
      
      def compute(self):
          self.p_out.set_from(self.p_in)  # assign values from `p_in`
          self.p_out.z = 0.0
  
  s = SomeSystem('s')
  # Multi-variable setter `set_values`
  s.p_in.set_values(x=1, y=-0.5, z=0.1)
  
  s.run_once()
  # Dict-like (key, value) iterator `items`:
  for varname, value in s.p_out.items():
      print(f"p_out.{varname} = {value})
  ```

### Documentation

* Updated tutorials on ports (MR [#139](https://gitlab.com/cosapp/cosapp/-/merge_requests/139)) and on optimization (MR [#140](https://gitlab.com/cosapp/cosapp/-/merge_requests/140)).

### Bug fixes and code quality

* Fix bug in Jacobian computation for negative perturbations (MR [#129](https://gitlab.com/cosapp/cosapp/-/merge_requests/129)).
* Fix bug in `RunOnce` and `RunSingleCase` recorders with `hold=False` (MR [#130](https://gitlab.com/cosapp/cosapp/-/merge_requests/130)).
* Resolve input aliasing in time driver scenarios (MR [#135](https://gitlab.com/cosapp/cosapp/-/merge_requests/135)).
* Fix bugs with events (MRs [#123](https://gitlab.com/cosapp/cosapp/-/merge_requests/123), [#126](https://gitlab.com/cosapp/cosapp/-/merge_requests/126), [#128](https://gitlab.com/cosapp/cosapp/-/merge_requests/128)).
* Discard empty connectors, and send a warning (MR [#132](https://gitlab.com/cosapp/cosapp/-/merge_requests/132)).
* Other code quality improvements (MRs [#127](https://gitlab.com/cosapp/cosapp/-/merge_requests/127), [#131](https://gitlab.com/cosapp/cosapp/-/merge_requests/131), [#133](https://gitlab.com/cosapp/cosapp/-/merge_requests/133), [#134](https://gitlab.com/cosapp/cosapp/-/merge_requests/134)).

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
