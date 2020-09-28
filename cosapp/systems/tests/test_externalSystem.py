from unittest import TestCase, skip

from cosapp.systems.externalsystem import ExternalSystem


class TestExternalSystem(TestCase):
    @skip("TODO")
    def test___init__(self):
        s = ExternalSystem("external")
        # self.s._options['command'].append(r'C:\Users\s554970\AppData\Local\Continuum\anaconda3\python')
        # self.s._options['command'].append(r'D:\users\s554970\Documents\TESTS\py_scripts\py.py')

        s._options["command"].append(
            r"C:\Appl\CAVIAR\CAVIAR_01.01.07\caviar\bin\win32\Python27\python"
        )
        # self.s._options['command'].append(r'D:\users\s554970\Documents\TESTS\caviarTools\caviar_launch.py')

        # self.s._execute_local()
        # self.s._process.wait()
        # self.s._process.communicate()

        # check_call(self.s._options['command'])

    @skip("TODO")
    def test___json__(self):
        self.fail()
