API references
==============

Modules
-------

.. autosummary::
   :toctree: _autosummary

   cosapp.core.numerics.basics
   cosapp.core.numerics.boundary
   cosapp.core.numerics.enum
   cosapp.core.numerics.residues
   cosapp.core.numerics.root
   cosapp.core.numerics.sobol_seq

   cosapp.core.signal.signal
   cosapp.core.signal.slot

   cosapp.core.config
   cosapp.core.eval_str
   cosapp.core.module
   cosapp.core.time

   cosapp.patterns.singleton
   cosapp.patterns.observer
   cosapp.patterns.visitor

   cosapp.ports.port
   cosapp.ports.enum
   cosapp.ports.connectors
   cosapp.ports.exceptions
   cosapp.ports.variable
   cosapp.ports.mode_variable
   cosapp.ports.units

   cosapp.systems.system
   cosapp.systems.systemfamily
   cosapp.systems.processsystem
   cosapp.systems.externalsystem
   cosapp.systems.metamodels

   cosapp.multimode.event
   cosapp.multimode.zeroCrossing
   cosapp.multimode.discreteStepper

   cosapp.drivers.driver
   cosapp.drivers.time.euler
   cosapp.drivers.time.interfaces
   cosapp.drivers.time.runge_kutta
   cosapp.drivers.time.scenario
   cosapp.drivers.time.utils
   cosapp.drivers.abstractsetofcases
   cosapp.drivers.abstractsolver
   cosapp.drivers.influence
   cosapp.drivers.iterativecase
   cosapp.drivers.lineardoe
   cosapp.drivers.metasystembuilder
   cosapp.drivers.montecarlo
   cosapp.drivers.nonlinearsolver
   cosapp.drivers.optimizer
   cosapp.drivers.optionaldriver
   cosapp.drivers.runonce
   cosapp.drivers.runsinglecase
   cosapp.drivers.validitycheck
   cosapp.drivers.utils

   cosapp.recorders.recorder
   cosapp.recorders.dataframe_recorder
   cosapp.recorders.dsv_recorder

   cosapp.tools.fmu.exporter
   cosapp.tools.problem_viewer.problem_viewer
   cosapp.tools.problem_viewer.webview
   cosapp.tools.help
   cosapp.tools.trigger
   cosapp.tools.module_parser.parseModule

   cosapp.utils.distributions.distribution
   cosapp.utils.distributions.normal
   cosapp.utils.distributions.triangular
   cosapp.utils.distributions.uniform
   cosapp.utils.swap_system

   cosapp.utils.context
   cosapp.utils.helpers
   cosapp.utils.json
   cosapp.utils.logging
   cosapp.utils.naming
   cosapp.utils.options_dictionary
   cosapp.utils.parsing
   cosapp.utils.pull_variables
   cosapp.utils.find_variables
   cosapp.utils.validate


Inheritance
-----------

- cosapp.core.numerics

.. mermaid-inheritance::
    cosapp.core.numerics.basics
    cosapp.core.numerics.boundary
    cosapp.core.numerics.enum
    cosapp.core.numerics.residues
    cosapp.core.numerics.root
    :parts: 1

- cosapp.drivers

.. mermaid-inheritance::
    cosapp.core.module
    cosapp.drivers.driver
    cosapp.drivers.abstractsetofcases
    cosapp.drivers.abstractsolver
    cosapp.drivers.influence
    cosapp.drivers.iterativecase
    cosapp.drivers.lineardoe
    cosapp.drivers.metasystembuilder
    cosapp.drivers.montecarlo
    cosapp.drivers.nonlinearsolver
    cosapp.drivers.optimizer
    cosapp.drivers.optionaldriver
    cosapp.drivers.runonce
    cosapp.drivers.runsinglecase
    cosapp.drivers.validitycheck
    cosapp.drivers.time.euler
    cosapp.drivers.time.interfaces
    cosapp.drivers.time.runge_kutta
    :parts: 1

- cosapp.ports

.. mermaid-inheritance::  
    cosapp.ports.port
    :parts: 1

- cosapp.ports.connectors

.. mermaid-inheritance::
    cosapp.ports.connectors
    :parts: 1

- cosapp.recorders

.. mermaid-inheritance::
    cosapp.recorders.recorder
    cosapp.recorders.dataframe_recorder
    cosapp.recorders.dsv_recorder
    :parts: 1

- cosapp.systems

.. mermaid-inheritance::
    cosapp.core.module
    cosapp.systems.system
    cosapp.systems.systemfamily
    cosapp.systems.metamodels
    cosapp.systems.externalsystem
    cosapp.systems.processsystem
    :parts: 1
