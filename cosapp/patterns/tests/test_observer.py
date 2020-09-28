"""Test observer pattern, consisting of classes `Subject` and `Observer`"""

import pytest

from cosapp.patterns import Subject, Observer


class DummyObserver(Observer):
    def _update(self):
        pass


class Counter(Observer):
    """Basic observer with internal counter incremented at each update"""
    def __init__(self, subject=None):
        super().__init__(subject)
        self.__count = 0

    def _update(self):
        self.__count += 1
    
    @property
    def count(self):
        return self.__count


@pytest.fixture(scope='function')
def subject():
    s = Subject()
    yield s
    assert s.n_observers == 0


def test_Counter(subject):
    counter = Counter()
    assert not counter.observes()
    assert not counter.observes(subject)

    counter = Counter(subject)
    assert counter.observes()
    assert counter.observes(subject)
    assert counter.count == 0

    counter = Counter()
    counter.observe(subject)
    assert counter.observes()
    assert counter.observes(subject)
    assert counter.count == 0
    new_subject = Subject()
    assert not counter.observes(new_subject)

    # first notification
    # at this point, subject has only one observer
    subject.notify()
    assert counter.count == 1
    
    other = Counter(subject)  # new observer
    assert other.observes(subject)
    
    for _ in range(4):
        subject.notify()
    assert counter.count == 5
    assert other.count == 4

    other.quit()
    assert not other.observes()
    subject.notify()
    assert counter.count == 6
    assert other.count == 4, "other's counter should not change, as observer has quit"


def test_observes():
    subjects = [Subject() for _ in range(2)]
    counter = Counter()
    assert not counter.observes()
    for i, subject in enumerate(subjects):
        assert not counter.observes(subject), f"with subject = subjects[{i}]"

    counter.observe(subjects[0])
    assert counter.observes()
    assert counter.observes(subjects[0])
    assert not counter.observes(subjects[1])

    counter.observe(subjects[1])
    assert counter.observes()
    assert not counter.observes(subjects[0])
    assert counter.observes(subjects[1])

    counter.quit()
    assert not counter.observes()
    assert not counter.observes(subjects[0])
    assert not counter.observes(subjects[1])


def test_Observer__del__():
    subject = Subject()
    assert subject.n_observers == 0

    counter = Counter(subject)
    assert counter.observes(subject)
    assert subject.n_observers == 1

    dummy = DummyObserver(subject)
    assert dummy.observes(subject)
    assert subject.n_observers == 2

    del counter
    assert subject.n_observers == 1
    del dummy
    assert subject.n_observers == 0


def test_Subject__del__():
    subject = Subject()
    assert subject.n_observers == 0

    dummy = DummyObserver(subject)
    counter = Counter(subject)
    assert subject.n_observers == 2
    assert dummy.observes(subject)
    assert counter.observes(subject)

    del subject
    assert not counter.observes()
    assert not dummy.observes()


def test_observer_type():
    assert issubclass(Subject.observer_type(), Observer)

    class DummySubject(Subject):
        def __new__(cls):
            return super().__new__(cls, DummyObserver)

    subject = DummySubject()
    assert issubclass(DummySubject.observer_type(), Observer)
    assert issubclass(DummySubject.observer_type(), DummyObserver)
    assert issubclass(subject.observer_type(), DummyObserver)

    observer = DummyObserver()
    observer.observe(subject)

    counter = Counter()
    with pytest.raises(TypeError):
        counter.observe(subject)

    with pytest.raises(TypeError):
        subject.add(counter)


def test_Subject_n_observers(subject):
    assert subject.n_observers == 0
    counters = [Counter(subject) for _ in range(4)]
    assert subject.n_observers == 4
    assert all(counter.observes(subject) for counter in counters)


def test_Subject_clear(subject):
    counters = [Counter(subject) for _ in range(4)]
    assert subject.n_observers == 4
    assert all(counter.observes(subject) for counter in counters)
    subject.clear()
    assert subject.n_observers == 0
    assert all(not counter.observes(subject) for counter in counters)
