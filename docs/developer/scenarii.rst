Sequence diagrams
-----------------

In this section some key scenarios will be described on a code perspective to help understanding the
call stacks. The illustrations are provided through UML sequence diagrams.


System initialization
~~~~~~~~~~~~~~~~~~~~~

When instantiating a ``System`` through code such as:

.. code-block:: python

    s = MySystem('main')

The following call stack is induced:

.. mermaid::

   sequenceDiagram
      participant main as main: System
      participant inwards as inwards: ExtensiblePort
      participant outwards as outwards: ExtensiblePort
      participant child as child: System
      participant in_ as in_: Port
      participant out as out: Port
      main->>+main: init()
      main->>inwards: <<create>> _add_port()
      main->>outwards: <<create>> _add_port()
      main->>main: _initialize()
      main->>+main: setup()
      main->>child: <<create>> add_child()
      main->>in_: <<create>> add_input()
      main->>out: <<create>> add_output()
      deactivate main
      main->>main: update()
      main->>+main: enforce_scope()
      main->>inwards: set scope_clearance
      main->>in_: set scope_clearance
      deactivate main
      deactivate main

The ``__init__`` method has the following actions:

1. Create class attributes needed by CoSApp (i.e. not the user 
   variables)
2. Creates the ports for the *inward* and the *outwards* through ``_add_port``.
3. ``_initialize`` method is called. By default this does nothing - its purpose is to provide an 
   hook method that allow child class to create *inward* and *outward* before the user *setup* is 
   called.
4. User defined ``setup`` is called. By default this does nothing. But in the diagram, the scenario
   presents a ``setup`` creating a *child* ``System``, an input ``Port`` and and output ``Port``.
5. ``update`` method is called. By default this does nothing. Its purpose is to build complex 
   objects from simple *inward*; for example reading a table. So this could be called manually when 
   updating the table filename for example.
6. The final step defines the scope of a variable depending on the user roles.

System simple execution
~~~~~~~~~~~~~~~~~~~~~~~

To execute once the ``compute`` method of a ``System``, you will use the method ``run_once``:

.. code-block:: python
   
    s = MySystem('main')
    s.run_once()

The call stack is as follow:

.. mermaid::

   sequenceDiagram
      participant main as main: System
      participant child as child: System
      participant connector as connector: Connector
      participant connector_out as connector_out: Connector
      main->>+main: run_once()
      main->>main: is_master_set()
      main->>+main: call_setup_run()
      main->>main: setup_run()
      main->>child: call_setup_run()
      deactivate main
      main->>main: _precompute()
      main->>main: compute_before()
      loop Every child
         loop Every connector on child inputs
            main->>connector: transfer()
         end
         main->>child: run_once()
      end
      loop Every connector pulling value from children
         main->>connector_out: transfer()
      end
      main->>main: compute()
      main->>main: _postcompute()
      main->>+main: call_clean_run()
      main->>main: clean_run()
      main->>child: call_clean_run()
      deactivate main
      deactivate main


1. The calling system test if it is the master one (i.e. is the one from which the execution call
   was emitted).
2. Through ``call_setup_run`` method, ``setup_run`` method of the system and all its children will
   be called recursively. By default ``setup_run`` does nothing. It is a hook to carry out operations
   needed only once per run. The method will only be executed by the master system.
3. Next comes the ``precompute`` method. 
4. Then comes the user hook ``compute_before``. This is the part of the compute that needs to be
   evaluated before the children.
5. Input ports of the child are initialized by pushing the value through the connectors.
   from their connector. If there is no connector, the value stays unchanged.
6. Method ``run_once`` of all children is called
7. The output ports of ``main`` are then updated. I.e. output ports connected to children output ports are
   transferred to the parent.
8. User defined ``compute`` is then called.
9. ``postcompute`` updates clean/dirty status and evaluates the system residues.
10. ``call_clean_run`` method is ending this scenario. This is undo ``setup_run`` actions following the user defined ``clean_run`` method. The method will only be executed by the master system.

Driver execution
~~~~~~~~~~~~~~~~

To execute a driver on a ``System``, you will use the method ``run_drivers`` after adding some
drivers to the system.

.. code-block:: python

   s = MySystem('main')
   s.add_driver(MyDriver('runner'))
   s.run_drivers()

The generic call stack is presented below. To simplify it the call to ``call_setup_run`` and
``call_clean_run`` have not been detailed.

.. mermaid::

   sequenceDiagram
      participant main as main: System
      participant runner as topRunner: Driver
      main->>+main: run_drivers()
      main->>main: is_master_set()
      main->>main: open_loops()
      main->>+main: call_setup_run()
      main->>runner: call_setup_run()
      deactivate main
      main->>runner: run_once()
      main->>+main: call_clean_run()
      main->>runner: call_clean_run()
      deactivate main
      main->>main: close_loops()
      deactivate main


There are two major differences with ``run_once``:

1. If the system is the master (i.e. the one calling the execution), ``open_loops`` method will 
   be executed to open connection loops in the system. This will be undone at the end through
   ``close_loops``.
2. The *compute* sequence is not called. Instead the ``run_once`` method of the drivers are called.

One call not shown above, is that the ``call_setup_run`` will trigger the ``setup_run`` method of 
the drivers to be called (similarly to the system children).

``NonLinearSolver`` Logic
^^^^^^^^^^^^^^^^^^^^^^^^^

As the major driver used in CoSApp is the ``NonLinearSolver``, the sequence diagram in that case is
described below for a system having one child and with only one ``RunSingleCase`` subdriver:


.. mermaid::

   sequenceDiagram
      autonumber
      participant main as main: System
      participant child as child: System
      participant solver as solver: NonLinearSolver
      participant point as point: RunSingleCase
      participant subproblem as subproblem: MathematicalProblem
      participant fullproblem as fullproblem: MathematicalProblem
      main->>+main: run_drivers()
      main->>main: is_master_set()
      main->>main: open_loops()
      main->>+main: call_setup_run()
      main->>+solver: call_setup_run()
      solver->>+solver: setup_run()
      solver->>+point: call_setup_run()
      point->>+point: setup_run()
      point->>subproblem: <<create>>
      point->>+main: assembled_problem()
      main->>+child: assembled_problem()
      child->>-main: child problem
      main->>-point: full unsolved problem
      point->>subproblem: set unknowns
      point->>subproblem: set equations
      deactivate point
      deactivate point
      deactivate solver
      main->>main: setup_run()
      deactivate main
      main->>+solver: run_once()
      solver->>+solver: _precompute()
      solver->>fullproblem: <<create>>
      loop RunSingleCase
         solver->>+point: get_problem()
         point->>-solver: single case problem
         solver->>+point: get_init()
         point->>-solver: single case initial values
      end
      deactivate solver
      solver->>+solver: compute_before()
      loop RunSingleCase
         solver->>point: set_iteratives()
      end
      deactivate solver
      loop Children drivers
         solver->>+point: run_once()
         point->>point: _precompute()
            Note right of point: Set boundaries and unknowns
         point->>+point: compute()
         point->>main: run_children_drivers()
         Note right of main: Initializing run
         deactivate point
         point->>point: _postcompute()
         deactivate point
      end
      solver->>+solver: compute()
      solver->>fullproblem: validate()
      solver->>+solver: resolution_method()
      loop Iterations
         solver->>+solver: fresidues()
         solver->>point: set_iteratives()
         solver->>+point: run_once()
         point->>point: _precompute()
         point->>+point: compute()
         point->>main: run_children_drivers()
         deactivate point
         point->>point: _postcompute()
         deactivate point
         deactivate solver
      end
      deactivate solver
      deactivate solver
      deactivate solver
      main->>+main: call_clean_run()
      main->>solver: call_clean_run()
      deactivate main
      main->>main: close_loops()
      deactivate main

The key elements are:

1. At ``call_setup_run`` execution, each ``RunSingleCase`` will create its own mathematical 
   problem (through calls 9 to 15). This is done by gathering the intrinsic mathematical problem of
   the system obtained with ``assembled_problem``. Then user defined unknowns and 
   equations will be added to build the mathematical problem on a single point.
2. During the ``precompute`` execution of ``NonLinearSolver``, the full mathematical problem will
   be built requesting the problem from each ``RunSingleCase`` (call 20 ``get_problem``).
3. With the call 26 ``run_once``, a simple run for initialization is called. This sets the 
   boundaries and the initial values.
4. Everything is now in place to solve the non-linear problem using one of the algorithms 
   available. This is initiated by call 33 ``resolution_method``. They all share a common interface 
   requesting a function returning the vector residues. That method called ``fresidues`` is defined in
   ``AbstractSolver``. It handle the dispatch of the unknowns values to the multiple ``RunSingleCase``
   as well as the execution of the multiple points to gather all residues required.


Transient Simulation
^^^^^^^^^^^^^^^^^^^^

Here is the sequence diagram for a transient simulation with explicit time integration.

.. mermaid::

   sequenceDiagram
      autonumber
      participant main as main: System
      participant child as child: System
      participant solver as solver: RungeKutta
      participant manager as TimeVarManager
      participant clock as UniversalClock

      main->>+main: run_drivers()
      main->>main: is_master_set()
      main->>main: open_loops()
      main->>+main: call_setup_run()
      main->>+solver: call_setup_run()
      solver->>+solver: setup_run()
      deactivate solver
      main->>main: setup_run()
      main->>manager: <<create>>
      manager->>+manager: update_transients()
      manager->>+main: assembled_problem()
      main->>+child: assembled_problem()
      child-->>-main: Child problem
      main-->>-manager: Full problem (including transients and rates)
         Note right of manager: Construct TransientUnknownStack
      manager->>clock: reset(start_time)
      deactivate manager
      deactivate main
      main->>+solver: run_once()
      solver->>solver: _precompute()
         Note right of solver: Solve children drivers
      solver->>+solver: compute()
         solver->>+solver: _initialize()
            solver->>+solver: _set_time(start_time)
            solver->>clock: time = t
            solver->>solver: Update boundary conditions
            solver->>solver: _update_children()
               Note right of solver: Solve children drivers
            solver->>solver: _update_rates()
         deactivate solver
         solver->>solver: Update initial conditions
         deactivate solver
         loop on Time
            solver->>+solver: _set_time(time)
               solver->>clock: time = t
               solver->>solver: Update boundary conditions
               solver->>solver: _update_children()
                  Note right of solver: Solve children drivers
               solver->>solver: _update_rates()
            deactivate solver
            solver->>solver: Get new time step
            solver->>solver: _update_transients(dt)
               Note right of solver: Time integration
         end
      deactivate solver
      deactivate solver
      deactivate solver
      main->>+main: call_clean_run()
      main->>solver: call_clean_run()
      deactivate main
      main->>main: close_loops()
      deactivate main
