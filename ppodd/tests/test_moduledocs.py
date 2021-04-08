import unittest

from ppodd.pod.base import pp_register
pp_modules = []
for key, value in pp_register.items():
    pp_modules += value

class TestDocStr(unittest.TestCase):
    """
    An empty subclass of unittest.TestCase, to which we append tests for each
    module.
    """

def add_test(module, func):
    """
    Add a test to TestMetaData, the unittest.TestCase subclass which contains
    all tests herein.
    """
    test_method = func(module)
    test_name = 'test_{}_{}'.format(module.__name__, func.__name__)
    test_method.__name__ = test_name
    setattr(TestDocStr, test_name, test_method)


def init_module(module):
    """
    Initialise a module inside a DecadesDataset, allowing its metadata to be
    interrogated.

    Args:
        module: a ppodd.pod postprocessing module.

    Returns:
        d: A DecadesDataset, in which module has been run.
    """

    _mod = module.test_instance()
    _mod.process()
    _mod.finalize()

    return _mod


def testdocstr(module):
    """
    Ensure that every module supplies test data for every required input.
    """

    def do_test(self):
        d = init_module(module)

        self.assertIsNotNone(d.__doc__)

    return do_test

# Add tests for each pp module
for module in pp_modules:
    add_test(module, testdocstr)
