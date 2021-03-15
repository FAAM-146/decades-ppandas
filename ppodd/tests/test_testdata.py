import unittest

from ppodd.pod import pp_modules

class TestTests(unittest.TestCase):
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
    setattr(TestTests, test_name, test_method)


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


def testdata(module):
    """
    Ensure that every module supplies test data for every required input.
    """

    def do_test(self):
        d = init_module(module)

        if callable(d.test):
            provided_test_data = d.test().keys()
        else:
            provided_test_data = d.test.keys()

        for _input in d.inputs:
            self.assertTrue(_input in provided_test_data)

    return do_test

# Add tests for each pp module
for module in pp_modules:
    add_test(module, testdata)
