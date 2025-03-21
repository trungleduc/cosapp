"""
Event handling and discrete stepping
"""
from __future__ import annotations
import scipy.optimize
import math

from typing import NamedTuple, Iterator, List, Dict, Tuple, Any, TYPE_CHECKING
from cosapp.multimode.event import Event, EventError
from cosapp.utils import partition
from cosapp.utils.state_io import object__getstate__
if TYPE_CHECKING:
    from cosapp.drivers import Driver
    from cosapp.drivers.time.utils import SystemInterpolator


class EventRecord(NamedTuple):
    """Named tuple associating a list of joint events
    and their occurrence time.
    """
    time: float
    events: List[Event]

    @classmethod
    def empty(cls):
        return cls(math.inf, [])


class DiscreteStepper():

    __slots__ = (
        '_owner', '_state', '_system', '_sysview',
        '_interval', '_nonprimitives', '_primitives',
    )

    def __init__(self, driver: Driver):
        # Local import to avoid cyclic dependency
        from cosapp.drivers.time.utils import SystemInterpolator

        self._owner = driver
        self._system = driver.owner
        self._sysview = SystemInterpolator(driver)
        self._interval = None
        self.update_events()

    def __json__(self) -> Dict[str, Any]:
        """Creates a JSONable dictionary representation of the object.

        Returns
        -------
        Dict[str, Any]
            The dictionary
        """
        _, slots = object__getstate__(self)
        slots.pop("_owner")
        slots.pop("_system")
        events = [pair for pair in self._state.items()]
        slots.update({"_state": events})
        return slots

    def update_sysview(self) -> None:
        """Refresh the view on the system of interest, in case the latter has changed."""
        self._sysview.refresh()

    def update_events(self) -> None:
        """Update event list from system of interest,
        and from stop criterion of owner time driver.
        """
        self._state: Dict[Event, bool] = dict.fromkeys(self._system.all_events(), False)
        stop = self._owner.scenario.stop
        if stop is not None:
            self._state[stop] = False
        primitives, non_primitives = partition(self.events(), lambda event: event.is_primitive)
        self._primitives: List[Event] = primitives
        self._nonprimitives: List[Event] = non_primitives

    def reset(self) -> None:
        """Update event list, and reset all events"""
        self.update_events()
        for event in self.events():
            event.reset()

    def initialize(self) -> None:
        """Update event list, and initialize all events"""
        self.update_events()
        for event in self.events():
            event.initialize()

    def events(self) -> Iterator[Event]:
        """Iterator on handled events"""
        return self._state.keys()

    def present_events(self) -> Iterator[Event]:
        """Returns an iterator on all present events"""
        return filter(lambda event: event.present, self.events())

    @property
    def sysview(self) -> SystemInterpolator:
        """SystemInterpolator: system interpolator used for event time finding"""
        return self._sysview

    @property
    def interval(self) -> Tuple[float, float]:
        """Tuple[float, float]: time interval"""
        return self._interval

    @interval.setter
    def interval(self, interval: Tuple[float, float]):
        self._interval = interval

    def set_data(self, interval, interpol) -> None:
        """Sets interpolation data used for detecting the first event that occurred in a given time step."""
        self.interval = interval
        self._sysview.interp = interpol

    def trigger_time(self, event: Event) -> float:
        """Returns the date at which a primitive event was triggered.
        This method may only be called once the list of all triggered events is known
        and the interpolation data has been set."""
        try:
            t_event = event._trigger_time()  # implemented if event is explicitly timed

        except EventError:
            # Event is not a timed event; find occurrence time by root finding
            sysview = self._sysview
            def f(t) -> float:
                sysview.exec(t)
                return event.value()
            t1, t2 = self._interval
            t_event = scipy.optimize.brentq(f, t1, t2)
            # Reset system at original state
            sysview.exec(t1)
        
        return t_event

    def find_primal_events(self) -> EventRecord:
        """Returns a `EventRecord` named tuple containing the first primitive events triggered,
        together with their occurrence time ((inf, []) if no primitive event is triggered).
        The internal state is only updated if an event is triggered.
        This method should only be called at the first microstep of the first discrete time step.
        """
        triggered_events = set(filter(lambda event: event.present, self._primitives))
        record = EventRecord.empty()
        for event in triggered_events:
            trigger_time = self.trigger_time(event)
            if math.isclose(trigger_time, record.time, rel_tol=1e-12):
                record.events.append(event)
            elif trigger_time < record.time:
                record = EventRecord(trigger_time, [event])
        if record.events:
            self._sysview.exec(record.time)
        for event in record.events:
            self._state[event] = True
        # Cancel primitive events that occured after the earliest detected
        for event in triggered_events.difference(record.events):
            event._cancel()
            event.reevaluate()
        return record

    def event_detected(self) -> bool:
        """Tests all primitive events and returns a Boolean value indicating
        whether at least one of these events is to be triggered."""
        return any(event.to_trigger() for event in self._primitives)

    def __microstep(self) -> bool:
        """Performs an evaluation microstep for non-primitive events and returns a Boolean
        indicating whether one or several events were triggered in the current cascade.
        This method should not be called at the first microstep of a discrete time step.
        """
        has_changed = False
        for event in self._nonprimitives:
            triggered = event.step()
            self._state[event] |= triggered
            has_changed |= triggered
        return has_changed

    def first_discrete_step(self) -> EventRecord:
        """Performs the first discrete step and returns an `EventRecord`
        indicating which primitive event was triggered and at which date."""
        for event in self._primitives:
            event.step()
        record = self.find_primal_events()
        while self.__microstep():
            pass
        return record

    def discrete_step(self) -> List[Event]:
        """Performs a discrete step other than the first one"""
        for event in self._primitives:
            self._state[event] = event.step()
        while self.__microstep():
            pass
        return list(self.present_events())

    def tick(self):
        """Ends a discrete loop by ticking all events."""
        for event in self.events():
            event.tick()

    def shift(self) -> None:
        """Performs a time shift of all primitive events."""
        for event in self._primitives:
            event.tick()

    def reevaluate_primitive_events(self) -> None:
        for event in self._primitives:
            event.reevaluate()
