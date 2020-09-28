"""This module test rapidly the time to set and get variable in `Port`
"""
from cosapp.ports import Port
from cosapp.ports.enum import PortType


class NumPort(Port):
    def setup(self):
        self.add_variable("Pt", 101325.0, dtype=float)
        self.add_variable("W", 1.0)


if __name__ == "__main__":
    from timeit import timeit

    n = 100000

    setup = (
        'from __main__ import NumPort, PortType; p = NumPort("myPort", PortType.IN);'
    )
    print("Time get:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit("p.Pt", setup=setup, number=n)
        )
    )

    setup += "from random import random"
    print("Time set with type checking:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit("p.W = random()", setup=setup, number=n)
        )
    )

    # No type checking
    Port.set_type_checking(False)
    print("Time set without type checking:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit("p.W = random()", setup=setup, number=n)
        )
    )

# Results for 7feafef6afcc1297bd88b8f4db1f8b10f83bf289
# Time get:
# 10000 loops, best of 3: 0.0218 ms per loop
# Time set:
# 10000 loops, best of 3: 0.216 ms per loop

# Results for 3a5cfc997b4e628d5c2f05b16c106426fee05ac9
# Time get:
# 100000 loops, best of 3: 0.0095 ms per loop
# Time set:
# 100000 loops, best of 3: 0.0205 ms per loop
