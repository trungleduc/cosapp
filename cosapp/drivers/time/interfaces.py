import abc
import logging
import numpy
from io import StringIO
from numbers import Number
from typing import Tuple, Union

from cosapp.core.time import UniversalClock
from cosapp.drivers.driver import Driver
from cosapp.drivers.time.utils import TimeVarManager, TimeStepManager
from cosapp.drivers.time.scenario import Scenario
from cosapp.recorders.recorder import BaseRecorder
from cosapp.utils.helpers import check_arg
from cosapp.utils.logging import LogFormat, LogLevel

logger = logging.getLogger(__name__)


class ExplicitTimeDriver(Driver):
    """
    Generic implementation of an explicit time driver with constant time step.
    Specialization of derived classes is achieved by the implementation of abstract method `_update_transients`
    """

    __slots__ = (
        '__time_interval', '__clock', '__recordPeriod',
        '_transients', '_rates', '__dt_manager', '__scenario',
        'record_dt', '__recorded_dt'
    )

    def __init__(self,
        name = "Explicit time driver",
        owner: "Optional[cosapp.systems.System]" = None,
        time_interval: Tuple[float, float] = None,
        dt: float = None,
        record_dt: bool = False,
        **options):
        """Initialization of the driver

        Parameters
        ----------
        name : str, optional
            Driver's name; default: "Explicit time driver"
        owner : System, optional
            :py:class:`~cosapp.systems.system.System` to which this driver belongs; default None
        time_interval : Tuple[float, float]
            Time interval [t_begin, t_end], with t_end > t_begin; defaut None
        dt : float
            Time step; defaut None. If None, will be tentatively determined
            from system transient variables.
        record_dt : bool
            If True, driver will store actual time steps used in simulation; default False.
            This option is only useful for post-run analysis, when `dt` is unspecified.
        **options : Dict[str, Any]
            Optional keywords arguments for generic `Driver` objects
        """
        dt_growth_rate = options.pop('max_dt_growth_rate', 2)
        super().__init__(name, owner, **options)
        self.__time_interval = None
        self.__recorded_dt = numpy.array([])
        self.__clock = UniversalClock()
        self.__recordPeriod = None
        self.__dt_manager = TimeStepManager(max_growth_rate=dt_growth_rate)
        self._transients = dict()
        self._rates = dict()
        self.dt = dt
        self.time_interval = time_interval
        self.record_dt = record_dt
        self.__scenario = Scenario("empty", self)

    def is_standalone(self) -> bool:
        return True

    @property
    def dt(self) -> Union[None, Number]:
        """Nominal time step (None if unspecified)"""
        return self.__dt_manager.nominal_dt

    @dt.setter
    def dt(self, value: Number) -> None:
        self.__dt_manager.nominal_dt = value

    @property
    def time(self) -> Number:
        """Current simulation time"""
        return self.__clock.time

    @property
    def time_interval(self) -> Tuple[Number, Number]:
        """Time interval covered by driver"""
        return self.__time_interval

    @time_interval.setter
    def time_interval(self, interval: Tuple[Number, Number]) -> None:
        if interval is not None:
            check_arg(interval, 'time_interval', (tuple, list), lambda it: len(it) == 2)
            interval = tuple(interval)
            start, end = interval
            check_arg(start, 'start time', Number, value_ok = lambda t: t >= 0)
            check_arg(end, 'end time', Number, value_ok = lambda t: t >= start and numpy.isfinite(t))
        self.__time_interval = interval

    # @property
    # def start_time(self):
    #     return self.__time_interval[0]

    # @property
    # def end_time(self):
    #     return self.__time_interval[1]

    def set_scenario(self, name="scenario", init=dict(), values=dict()) -> None:
        """
        Define a simulation scenario, from initial and boundary conditions.

        Parameters
        ----------
        init : dict
            Dictionary of initial conditions, of the kind {variable: value}
        values : dict
            Dictionary of boundary conditions, of the kind {variable: value}
            Explicit time dependency may be specified, as {variable: 'cos(omega * t)'}, for example
        name : str
            Name of the scenario (by default, 'scenario')
        """
        self.__scenario = Scenario.make(name, init, values, self)

    @property
    def scenario(self) -> Scenario:
        """Scenario: the simulation scenario, defining initial and boundary conditions"""
        return self.__scenario

    def add_recorder(self, recorder: BaseRecorder, period: Number = None) -> BaseRecorder:
        """Add an internal data recorder storing the time evolution of values of interest"""
        if 'time' not in recorder:
            cls = type(recorder)
            recorder = cls.extend(recorder, includes='time')
        super().add_recorder(recorder)
        if period is not None:
            self.recording_period = period
        return self.recorder

    def setup_run(self) -> None:
        """Setup the driver once before starting the simulation and before
        calling the systems `setup_run`.
        """
        if self.__time_interval is None:
            raise ValueError("Time interval was not specified")

        manager = TimeVarManager(self.owner)
        self.__dt_manager.transients = manager.transients
        self._transients = manager.transients
        self._rates = manager.rates
        logger.debug(f"Transient variables: {self._transients!r}")
        logger.debug(f"Rate variables: {self._rates!r}")
        self.__reset_time()

    def compute(self) -> None:
        """Simulate the time-evolution of owner System over a prescribed time interval"""
        self._initialize()

        t0, tn = self.__time_interval
        dt_manager = self.__dt_manager
        dt = self.dt or dt_manager.time_step()
        record_all = False
        t_record = numpy.inf

        if self._recorder is None:
            must_record = lambda t, t_record: False
            record = lambda : None
        else:
            if self.recording_period is None:
                if self.dt is None:
                    logger.warning(
                        "Unspecified recording period and time step"
                        "; all time steps will be recorded"
                    )
                record_all = True
            eps = min(1e-8, dt / 100)
            must_record = lambda t, t_record: abs(t - t_record) < eps
            record = lambda : self._recorder.record_state(
                float(f"{self.time:.14e}"), self.status, self.error_code)
            t_record = numpy.inf if record_all else t0

        recorded_dt = []
        if self.record_dt:
            record_dt = lambda dt: recorded_dt.append(dt)
        else:
            record_dt = lambda dt: None
        
        t = t0
        n_record = 0
        prev_dt = None

        while True:  # time loop
            self._set_time(t)
            if record_all:
                record()
            elif must_record(t, t_record):
                record()
                n_record += 1
                t_record = t0 + n_record * self.recording_period
            dt = dt_manager.time_step(prev_dt)
            next_t = t + dt
            # Update previous dt, unless current dt is artificially
            # limited by recording timestamp `t_record`
            if next_t > t_record:
                next_t = t_record
                dt = next_t - t
            else:
                prev_dt = dt
            if next_t > tn:
                break
            self._update_transients(dt)
            record_dt(dt)
            t = next_t

        remaining_dt = tn - t
        if remaining_dt > 1e-3 * dt:
            self._update_transients(remaining_dt)
            record_dt(remaining_dt)
            self._set_time(tn)
            record()

        self.__recorded_dt = numpy.asarray(recorded_dt)

    def _set_time(self, t: Number) -> None:
        dt = t - self.time
        self.__clock.time = t
        self.__scenario.update_values()
        self._update_children()
        self._update_rates(dt)

    def __reset_time(self) -> None:
        """Reset clock time to driver start time"""
        time = self.__time_interval[0]
        logger.debug(f"Reset time to {time}")
        self.__clock.reset(time)

    def _initialize(self):
        self.__reset_time()
        self.__scenario.apply_init_values()
        self.__scenario.update_values()
        for transient in self._transients.values():
            # re-synch stacked unknowns with root variables
            transient.reset()
        logger.debug("Reset rates")
        for rate in self._rates.values():
            rate.reset()

    def _update_children(self) -> None:
        if len(self.children) > 0:
            for name in self.exec_order:
                self.children[name].run_once()
        else:
            self.owner.run_children_drivers()

    def _update_rates(self, dt: Number) -> None:
        """Update rate-of-changes over time interval dt"""
        if dt == 0:
            return
        synch_needed = False
        for rate in self._rates.values():
            rate.update(dt)
            synch_needed = True
        if synch_needed:
            # Re-run dynamic system with updated (synchronized) rates
            # Equivalent to a single step fixed-point solver
            self._update_children()

    @abc.abstractmethod
    def _update_transients(self, dt: Number) -> None:
        """
        Time integration of transient variables over time step `dt`.
        Actual implementation depends on chosen numerical scheme.
        """
        pass

    @property
    def recording_period(self) -> Number:
        """float: Recording period of time driver's internal recorder"""
        return self.__recordPeriod

    @recording_period.setter
    def recording_period(self, value: Number):
        check_arg(value, 'recording_period', Number, lambda t: t > 0)
        interval = self.time_interval
        if interval is None:
            self.__recordPeriod = value
        else:
            self.__recordPeriod = min(value, interval[1] - interval[0])

    @property
    def recorded_dt(self) -> numpy.ndarray:
        """numpy.ndarray: list of time steps recorded during driver execution"""
        return self.__recorded_dt

    def log_debug_message(self, handler: "HandlerWithContextFilters", record: logging.LogRecord, format: LogFormat = LogFormat.RAW) -> bool:
        """Callback method on the system to log more detailed information.
        
        This method will be called by the log handler when :py:meth:`~cosapp.utils.logging.LoggerContext.log_context`
        is active if the logging level is lower or equals to VERBOSE_LEVEL. It allows
        the object to send additional log message to help debugging a simulation.

        Parameters
        ----------
        handler : HandlerWithContextFilters
            Log handler on which additional message should be published.
        record : logging.LogRecord
            Log record
        format : LogFormat
            Format of the message

        Returns
        -------
        bool
            Should the provided record be logged?
        """
        message = record.getMessage()
        activate = getattr(record, "activate", None)
        emit_record = super().log_debug_message(handler, record, format)

        if message.endswith("call_setup_run") or message.endswith("call_clean_run"):
            emit_record = False

        elif activate == True:
            self.record_dt = True
            emit_record = False

        elif activate == False:
            self.record_dt = False
            emit_record = False
            container = StringIO()
            numpy.savetxt(container, self.recorded_dt, delimiter=",")
            dts = container.getvalue()
            
            message = f"Time steps:\n{dts}"
            handler.log(
                LogLevel.FULL_DEBUG,
                message,
                name=logger.name,
            )

        return emit_record
