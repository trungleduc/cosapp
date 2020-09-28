"""This module test rapidly the time to set and get variable in `Port`
"""
from cosapp.ports import Port
from cosapp.systems import System


class NumPort(Port):
    def setup(self):
        self.add_variable("Pt", 101325.)
        self.add_variable("W", 1.)


class SubSystem(System):
    def setup(self):

        self.add_input(NumPort, "in_")
        self.add_inward("sloss", 0.95)
        self.add_output(NumPort, "out")

    def compute(self):
        for name in self.out:
            self.out[name] = self.in_[name] * self.inwards.sloss


class TopSystem(System):
    def setup(self):

        self.add_input(NumPort, "in_")
        self.add_output(NumPort, "out")
        self.add_child(SubSystem("sub"))

        self.connect(self.in_, self.sub.in_)
        self.connect(self.out, self.sub.out)


if __name__ == "__main__":
    from timeit import timeit

    n = 100000

    setup = 'from __main__ import TopSystem; s = TopSystem("mySystem"); s.run_once();'
    print("Time get s.out.Pt:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit("s.out.Pt", setup=setup, number=n)
        )
    )
    print("Time get s['out.Pt']:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit('s["out.Pt"]', setup=setup, number=n)
        )
    )
    print("Time get s.sub.inwards.sloss:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit("s.sub.inwards.sloss", setup=setup, number=n)
        )
    )
    print("Time get s['sub.inwards.sloss']:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit('s["sub.inwards.sloss"]', setup=setup, number=n)
        )
    )

    setup += "from random import random"
    print("Time set s.out.Pt:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit("s.out.Pt = random()", setup=setup, number=n)
        )
    )
    print('Time set s["out.Pt"]:')
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit('s["out.Pt"] = random()', setup=setup, number=n)
        )
    )
    print("Time set s.sub.inwards.sloss:")
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit("s.sub.inwards.sloss = random()", setup=setup, number=n)
        )
    )
    print('Time set s["sub.inwards.sloss"]:')
    print(
        "{} loops, best of 3: {:.3g} ms per loop".format(
            n, timeit('s["sub.inwards.sloss"] = random()', setup=setup, number=n)
        )
    )

# Results for 7feafef6afcc1297bd88b8f4db1f8b10f83bf289
# Time get s.out.Pt:
# 100000 loops, best of 3: 0.864 ms per loop
# Time get s.sub.inwards.sloss:
# 100000 loops, best of 3: 1.47 ms per loop
# Time set s.out.Pt:
# 100000 loops, best of 3: 2.93 ms per loop
# Time set s.sub.inwards.sloss:
# 100000 loops, best of 3: 3.32 ms per loop

# Results for 21652dfb46e5d1a553060a3a7e5229696c5db620
# Time get s.out.Pt:
# 100000 loops, best of 3: 0.417 ms per loop
# Time get s.sub.inwards.sloss:
# 100000 loops, best of 3: 1.06 ms per loop
# Time set s.out.Pt:
# 100000 loops, best of 3: 0.556 ms per loop
# Time set s.sub.inwards.sloss:
# 100000 loops, best of 3: 1.14 ms per loop

# Results for 67bc6594d29f580d336fb27740b48fe667ee4067
# Time get s.out.Pt:
# 100000 loops, best of 3: 0.57 ms per loop
# Time get s['out.Pt']:
# 100000 loops, best of 3: 0.346 ms per loop
# Time get s.sub.inwards.sloss:
# 100000 loops, best of 3: 1.26 ms per loop
# Time get s['sub.inwards.sloss']:
# 100000 loops, best of 3: 0.374 ms per loop
# Time set s.out.Pt:
# 100000 loops, best of 3: 0.659 ms per loop
# Time set s["out.Pt"]:
# 100000 loops, best of 3: 0.404 ms per loop
# Time set s.sub.inwards.sloss:
# 100000 loops, best of 3: 1.28 ms per loop
# Time set s["sub.inwards.sloss"]:
# 100000 loops, best of 3: 0.495 ms per loop

# Results for using try-except and avoiding call __g/setitem__ in __g/setattr__
# Time get s.out.Pt:
# 100000 loops, best of 3: 0.326 ms per loop
# Time get s['out.Pt']:
# 100000 loops, best of 3: 0.332 ms per loop
# Time get s.sub.inwards.sloss:
# 100000 loops, best of 3: 0.741 ms per loop
# Time get s['sub.inwards.sloss']:
# 100000 loops, best of 3: 0.373 ms per loop
# Time set s.out.Pt:
# 100000 loops, best of 3: 0.393 ms per loop
# Time set s["out.Pt"]:
# 100000 loops, best of 3: 0.417 ms per loop
# Time set s.sub.inwards.sloss:
# 100000 loops, best of 3: 0.757 ms per loop
# Time set s["sub.inwards.sloss"]:
# 100000 loops, best of 3: 0.415 ms per loop
