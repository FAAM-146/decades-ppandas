import datetime
import unittest

from lxml import etree
import requests
import pandas as pd

from cfunits import Units

from ppodd.decades import DecadesDataset, DecadesVariable
from ppodd.pod.base import pp_register
pp_modules = []
for key, value in pp_register.items():
    pp_modules += value


STANDARD_NAMES_URL = ('http://cfconventions.org/Data/cf-standard-names'
                      '/69/src/cf-standard-name-table.xml')

class TestMetaData(unittest.TestCase):
    """
    An empty subclass of unittest.TestCase, to which we append tests for each
    module
    """


class SilentDict(dict):
    """
    Subclass dict so repr doesn't print the whole dict and spam the test
    output.
    """
    def __repr__(self):
        return '<standard_names>'


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

    return _mod.dataset


def get_standard_names():
    """
    Pull and parse a list of cf standard names.

    Returns:
        standard_names: a dict of cf standard names, retreived from
            STANDARD_NAMES_URL, mapping standard names to canonical units.
    """

    standard_names = SilentDict()
    response = requests.get(STANDARD_NAMES_URL)

    tree = etree.fromstring(response.content)

    nodes = tree.findall('entry')
    for node in nodes:
        _std_name = node.attrib['id']
        _can_units = node.find('canonical_units').text

        standard_names[_std_name] = _can_units

    nodes = tree.findall('alias')
    for node in nodes:
        _alias = node.attrib['id']
        _entry = node.find('entry_id').text
        _can_units = standard_names[_entry]

        standard_names[_alias] = _can_units

    return standard_names


def add_test(module, func):
    """
    Add a test to TestMetaData, the unittest.TestCase subclass which contains
    all tests herein.
    """
    test_method = func(module)
    test_name = 'test_{}_{}'.format(module.__name__, func.__name__)
    test_method.__name__ = test_name
    setattr(TestMetaData, test_name, test_method)


def long_name(module):
    """
    Ensure that all output variables have been given a long_name attribute.

    Args:
        module: a ppodd.pod postprocessing module

    Returns:
        the test
    """
    def do_test(self):
        d = init_module(module)
        for var in d.outputs:
            self.assertTrue(hasattr(var, 'long_name'))
    return do_test


def valid_units(module):
    """
    Ensure all units attitbutes are valid, as specified by udunits.

    Args:
        module: a ppodd.pod postprocessing module

    Returns:
        the test
    """
    def do_test(self):
        d = init_module(module)
        for var in d.outputs:
            if hasattr(var, 'units'):
                _unit = Units(var.units)
                self.assertTrue(_unit.isvalid)
    return do_test

def valid_standard_name(module):
    """
    Ensure all standard names are cf-compliant.

    Args:
        module: a ppodd.pod postprocessing module

    Returns:
        the test
    """
    def do_test(self):
        d = init_module(module)
        for var in d.outputs:
            if (hasattr(var, 'standard_name')
                    and var.standard_name is not None):
                self.assertIn(
                    var.standard_name, names
                )
    return do_test

def valid_units_for_name(module):
    """
    Ensure units, if given, are equivalent to canonical units of standard name,
    if given.

    Args:
        module: a ppodd.pod postprocessing module

    Returns:
        the test
    """
    def do_test(self):
        d = init_module(module)
        for var in d.outputs:
            if not (hasattr(var, 'standard_name')
                        and hasattr(var, 'units')):
                continue

            output_units = Units(var.units)

            try:
                canonical_units = Units(
                    names[var.standard_name]
                )
            except KeyError:
                # Invalid standard name
                continue

            self.assertTrue(output_units.equivalent(canonical_units))
    return do_test

# Get a dict of standard names
names = get_standard_names()

# Add tests for each pp module
for module in pp_modules:
    add_test(module, long_name)
    add_test(module, valid_units)
    add_test(module, valid_standard_name)
    add_test(module, valid_units_for_name)
