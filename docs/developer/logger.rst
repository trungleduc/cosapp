Logger
------

CoSApp defines its own logger handler on top of Python `logging <https://docs.python.org/3/howto/logging.html>`_
package. The reason behind the definition of customized handlers is to invert the *logging* logic for fine
debugging. Meaning, instead of having log message built by CoSApp objects at all times and then filter by the logging
system depending on the log level, for debug level, the log handler will request CoSApp objects to provide
more information. This allows to reduce the execution overhead of producing debug messages when it is
not required.


First, the log message flow will be described (the figures below are reproduced from `Logger flow <https://docs.python.org/3/howto/logging.html#logging-flow>`_).
As seen in the flow graph below, when the code is calling the logger to emit a message, first the logger
test if the message should be considered depending on its level. Then if it does, a ``LogRecord`` object
is instantiated and logger filters are applied on it. And only then, is the record pass to the handlers.


Finally the flow is looping over the logger hierarchy.


.. mermaid::

   graph TD
      style startlog fill:#00000000,stroke:#00000000,color:#00000000
      style handlerflow fill:#00000000,stroke:#00000000,color:#00000000
      logenable{Logger enabled<br />for level of call?}
      create[Create LogRecord]
      rejected{Does a filter attached<br />to logger reject the<br />record?}
      tocurrent[Pass to handlers<br />of current logger]
      propagate{Is propagate true<br />for current logger?}
      parent{Is there a<br />parent logger?}
      setcurrent[Set current<br />logger to parent]
      stop([Stop])

      startlog-->|Logging call<br />in user code|logenable
      logenable-->|No|stop
      logenable-->|Yes|create
      create-->rejected
      rejected-->|Yes|stop
      rejected-->|No|tocurrent
      tocurrent-->propagate
      propagate-->|No|stop
      propagate-->|Yes|parent
      parent-->|No|stop
      parent-->|Yes|setcurrent
      setcurrent-->tocurrent

      tocurrent-. LogRecord passed to handler .->handlerflow

For each handler, specific debug level and filters can be defined. So the record is again tested against
those selection criteria. And if those criteria are respected the record is emitted on the proper support.

.. mermaid::

   graph TD
      style tocurrent  fill:#00000000,stroke:#00000000,color:#00000000
      handlerenable{Handler enabled for<br />level of LogRecord}
      handlerfilter{Does a filter attached<br />to handler reject<br />the record?}
      emit(["Emit (includes formatting)"])
      handlerstop([Stop])

      tocurrent-. LogRecord passed to handler .->handlerenable
      handlerenable-->|No|handlerstop
      handlerenable-->|Yes|handlerfilter
      handlerfilter-->|Yes|handlerstop
      handlerfilter-->|No|emit


To extend the logging Python system, the easiest method is to create filters and handlers. Both can then be added
to loggers - in particular to the root logger to catch all messages.

The other need is to get the active CoSApp context to let the handler know that it can request additional information.
This is done by passing the CoSApp object as extra attribute of a ``LogRecord`` (see :py:meth:`~cosapp.utils.logging.LoggerContext.log_context`).
Thanks to the context manager :py:meth:`~cosapp.utils.logging.LoggerContext.log_context`, you can in CoSApp code activate
for a method execution the context. For example, in :py:meth:`cosapp.core.module.Module.call_setup_run`:

.. code:: python

    def call_setup_run(self):
        """Execute `setup_run` recursively on all modules."""
        with self.log_context(" - call_setup_run"):
            logger.debug(f"Call {self.name}.setup_run")
            self._compute_calls = 0  # Reset the counter
            self.setup_run()
            for child in self.children.values():
                child.call_setup_run()
            self.setup_ran.emit()

To use it, the CoSApp class needs to inherit from :py:class:`~cosapp.utils.logging.LoggerContext` (this is the case of
:py:class:`~cosapp.core.module.Module` and therefore of :py:class:`~cosapp.drivers.driver.Driver` and :py:class:`~cosapp.systems.system.System`).
Then the convention used is to pass a string with the calling method. When entering **and** exiting the context manager,
the method ``log_debug_message`` of the context object will be called (if the log level is ``DEBUG`` or lower).
As a consequence to provide more information, that method needs to be overridden in the CoSApp object to be meaningful.
For example, in :py:class:`~cosapp.systems.system.System` ``.`` :py:meth:`~cosapp.systems.system.System.log_debug_message`,
no additional information is displayed for ``call_setup_run`` and ``call_clean_run``. But for ``run_once`` or ``run_driver``,
the inputs are displayed at the entering of the log context and the outputs at its exit.


.. note::

   As ``log_debug_message`` is called by the handler, additional information should be logged directly on it. So the handler
   is an argument of ``log_debug_message``. And logging a message can be done using the helper :py:meth:`~cosapp.utils.logging.HandlerWithContextFilters.log`.

   A drawback of the current approach is that ``log_debug_message`` will be called by all handlers.


The other advantage to pass the context to the handler allows to pass it to the filters. This is used by :py:class:`~cosapp.utils.logging.TimeFilter`
and :py:class:`~cosapp.utils.logging.ContextFilter`. The first one needs the context to see if its time is valid when the second
filters the context itself.


.. note::

   As the context is only propagated when the log level is lower or equal to ``DEBUG``, those two filters
   are accepting any log message of higher level than ``DEBUG``.


To clarify the above description, two sequence diagrams are shown next. The first one displays what happen when the log level is
greater than ``DEBUG`` (e.g. ``INFO``).


.. mermaid::

   sequenceDiagram
      autonumber
      participant main as main: System
      participant llogger as System logger
      participant logger as root logger
      participant handler as handler: FileLogHandler

      main->>+main: run_once()
      main->>+main: log_context(" - run_once")
      main->>+llogger: log(..." - run_once", extra{"activate": True, "context": self})
      llogger->>+logger: propagate LogRecord
         Note right of logger: LogRecord filtered out by log level
      deactivate logger
      deactivate llogger
         Note over main: Compute call logic
      main->>+llogger: log(..." - run_once", extra{"activate": False, "context": self})
      llogger->>+logger: propagate LogRecord
         Note right of logger: LogRecord filtered out by log level
      deactivate logger
      deactivate llogger
      deactivate main
      deactivate main


At the entering and exiting :py:meth:`~cosapp.utils.logging.LoggerContext.log_context`,
the log message carrying the context will never reach the handler. Therefore additional
information won't be requested.


Then the case of a *System* simulation with ``FULL_DEBUG`` is presented.


.. mermaid::

   sequenceDiagram
      autonumber
      participant main as main: System
      participant llogger as System logger
      participant logger as root logger
      participant handler as handler: FileLogHandler
      participant filter as ctxfilter: ContextFilter

      main->>+main: run_once()
      main->>+main: log_context(" - run_once")
      main->>+llogger: log(..." - run_once", extra{"activate": True, "context": self})
      llogger-->>+logger: propagate LogRecord
      logger->>+handler: handle(record)
      handler->>+handler: needs_handling
      handler->>filter: set context
      handler->>+main: log_debug_message(handler, record)
      main->>handler: log()
         Note right of handler: Log system inputs
      main-->>-handler: <<returns>>
      deactivate handler
      deactivate handler
      deactivate logger
      deactivate llogger
         Note over main: Compute call logic
      main->>+llogger: log(..." - run_once", extra{"activate": False, "context": self})
      llogger-->>+logger: propagate LogRecord
      logger->>+handler: handle(record)
      handler->>+handler: needs_handling
      handler->>filter: set context
      handler->>+main: log_debug_message(handler, record)
      main->>handler: log()
         Note right of handler: Log system outputs
      main-->>-handler: <<returns>>
      deactivate handler
      deactivate handler
      deactivate logger
      deactivate llogger
      deactivate main
      deactivate main
