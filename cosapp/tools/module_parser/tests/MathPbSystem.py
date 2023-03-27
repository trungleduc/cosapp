from cosapp.base import System

class SysWithUnknown(System):
    def setup(self):
        self.add_inward('x')
        self.add_unknown('x')


class SysWithEquation(System):
    def setup(self):
        self.add_outward('z')
        self.add_equation('z == [0, 0, 0]')
        
        
class AssemblyWithMathPb(System):
    def setup(self):
        self.add_child(SysWithUnknown('unk'))
        self.add_child(SysWithEquation('equa'))