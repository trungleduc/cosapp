"""Originally taken from http://code.activestate.com/recipes/576696/"""

import collections
from weakref import proxy


class Link:
    __slots__ = 'prev', 'next', 'elem', '__weakref__'


class OrderedSet(collections.abc.MutableSet):
    """A set of unique items that preserves the insertion order.

    Created by: Raymond Hettinger
    Version: 2009-03-19
    Licence: http://opensource.org/licenses/MIT
    Source: http://code.activestate.com/recipes/576694/

    An ordered set is implemented as a wrapper class for a dictionary
    implementing a doubly linked list. It also has a pointer to the last item
    in the set (self.end) which is used by the add and _iterate methods to add 
    items to the end of the list and to know when an iteration has finished,
    respectively."""
    # Big-O running times for all methods are the same as for regular sets.
    # The internal self.__map dictionary maps elems to links in a doubly linked list.
    # The circular doubly linked list starts and ends with a sentinel element.
    # The sentinel element never gets deleted (this simplifies the algorithm).
    # The prev/next links are weakref proxies (to prevent circular references).
    # Individual links are kept alive by the hard reference in self.__map.
    # Those hard references disappear when a elem is deleted from an OrderedSet.

    def __init__(self, iterable=None):
        self.__root = root = Link()  # sentinel node for doubly linked list
        root.prev = root.next = root
        self.__map = {}  # elem --> link
        if iterable is not None:
            self |= iterable

    def __len__(self) -> int:
        return len(self.__map)

    def __contains__(self, elem):
        return elem in self.__map

    def add(self, elem):
        """Append element as a new link at the end of the linked list"""
        if elem not in self.__map:
            self.__map[elem] = link = Link()
            root = self.__root
            last = root.prev
            link.prev, link.next, link.elem = last, root, elem
            last.next = root.prev = proxy(link)

    def append(self, elem):
        """Append an element to the set (mimicks the behaviour of list.append)."""
        self.add(elem)

    def index(self, value):
        """Returns the position of `elem` (mimicks the behaviour of list.index)."""
        for i, elem in enumerate(self):
            if elem == value:
                return i
        raise ValueError("{} is not in set".format(value))

    def discard(self, elem):
        """Remove an element from the set"""   
        if elem in self.__map:
            link = self.__map.pop(elem)
            link.prev.next = link.next
            link.next.prev = link.prev

    def insert(self, index, elem):
        """Store new element at a given position in the linked list"""
        if elem in self.__map:
            return
        index = self._insertion_index(index)
        curr = self.__root
        for i in range(index):
            curr = curr.next
        self.__map[elem] = link = Link()
        link.prev, link.next, link.elem = curr, curr.next, elem
        curr.next = proxy(link)

    def __iter__(self):
        """Iterate over the linked list in order."""
        root = self.__root
        curr = root.next
        while curr is not root:
            yield curr.elem
            curr = curr.next

    def __reversed__(self):
        """Iterate over the linked list in reverse order."""
        root = self.__root
        curr = root.prev
        while curr is not root:
            yield curr.elem
            curr = curr.prev

    def pop(self, last=True):
        """Pop the last element of the set if `last` is True.
        Otherwise, pop the first element."""
        if not self:
            raise KeyError('set is empty')
        elem = next(reversed(self)) if last else next(iter(self))
        self.discard(elem)
        return elem

    def __repr__(self):
        elems = list(self) if self else ""
        return "{}({})".format(self.__class__.__name__, elems)

    def __eq__(self, other):
        return isinstance(other, OrderedSet) and list(self) == list(other)

    def _insertion_index(self, index: int) -> int:
        """Get insertion position in the linked list"""
        if not isinstance(index, int):
            raise TypeError("insertion index must be int")
        n = len(self)
        index = min(index, n)
        if index < 0:
            index = n + max(index, -n)
        return index

    @property
    def first(self):
        try:
            return self.__root.next.elem
        except AttributeError:
            raise AttributeError("empty set")

    @property
    def last(self):
        try:
            return self.__root.prev.elem
        except AttributeError:
            raise AttributeError("empty set")
