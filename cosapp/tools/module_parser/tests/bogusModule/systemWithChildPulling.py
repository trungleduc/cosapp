from cosapp.base import System

from .systemAndPort import BogusSystem

class BogusSystemChildPulling(System):
    def setup(self) -> None:
        self.add_child(BogusSystem('bog'), pulling={'p_in': 'p_in_parent'})