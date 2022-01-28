import unittest
import logging

from typing import Optional

import pandas as pd

from pydantic import BaseModel, Field

import ppodd.readers as r
from ppodd.decades import DecadesDataset, DecadesVariable
from ppodd.decades.attributes import AttributesCollection, Attribute
from ppodd.decades.attributes import NonStandardAttributeError
from ppodd.decades.attributes import STR_DERIVED_FROM_FILE, ATTRIBUTE_NOT_SET

logging.disable(logging.CRITICAL)

class Attributes(BaseModel):
    Required: str = Field(description='A required attribute', example='my attribute')
    RequiredDefault: str = Field(
        description='A required attribute w/ default',
        ppodd_default='A required attribute'
    )
    OptionalAttr: Optional[str] = 'An optional attribute'

# test_definition = {
#     'Required': {
#         'required': True,
#         'description': 'A required attribute',
#         'aliases': [],
#         'versions': [1.0],
#         'inherits_from': None
#     },
#     'RequiredDefault': {
#         'required': True,
#         'description': 'A required attribute',
#         'aliases': [],
#         'versions': [1.0],
#         'inherits_from': None,
#         'default': 42
#     },
#     'Optional': {
#         'required': False,
#         'description': 'A required attribute',
#         'aliases': [],
#         'versions': [1.0],
#         'inherits_from': None
#     },
#     'key1': {
#         'required': False,
#         'description': 'A test attribute',
#         'aliases': [],
#         'versions': [1.0],
#         'inherits_from': None
#     },
#     'key2': {
#         'required': False,
#         'description': 'A test attribute',
#         'aliases': [],
#         'versions': [1.0],
#         'inherits_from': None
#     }
# }

DEF_PATH = 'ppodd.tests.test_attributes.Attributes'

class TestAttributes(unittest.TestCase):
    """
    An empty subclass of unittest.TestCase, to which we append tests for each
    module.
    """

    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def teardownClass(cls):
        pass

    def test_create_collection_with_no_definition(self):
        a = AttributesCollection()

    def test_create_collection_with_path_definition(self):
        a = AttributesCollection(definition=DEF_PATH)

    def test_required_attributes_from_definition(self):
        a = AttributesCollection(definition=DEF_PATH,)
        for key in ['Required', 'RequiredDefault']:
            self.assertIn(key, a.REQUIRED_ATTRIBUTES)

    def test_option_attributes_from_definition(self):
        a = AttributesCollection(definition=DEF_PATH)
        self.assertIn('OptionalAttr', a.OPTIONAL_ATTRIBUTES)

    def test_required_attributes_with_no_definition(self):
        a = AttributesCollection()
        self.assertEquals(a.REQUIRED_ATTRIBUTES, [])

    def test_optional_attributes_with_no_definition(self):
        a = AttributesCollection()
        self.assertEquals(a.OPTIONAL_ATTRIBUTES, [])

    def test_get_item_non_compliancify(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key2', 'value2'))
        self.assertEquals(a['key1'], 'value1')
        self.assertEquals(a['key2'], 'value2')

    def test_add_duplicate_key(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key1', 'value2'))
        self.assertEquals(a['key1'], 'value2')

    def test_add_invalid_key_in_strict_mode(self):
        a = AttributesCollection(definition=Attributes)
        self.assertRaises(
            NonStandardAttributeError,
            lambda: a.add(Attribute('invalid', 'invalid'))
        )

    def test_set_item(self):
        a = AttributesCollection()
        a['key1'] = 'value'
        self.assertEquals(a['key1'], 'value')

    def test_set_item_as_dict(self):
        a = AttributesCollection()
        a['key'] = {'level1': 'value1', 'level2': 'value2'}
        self.assertEquals(a['key_level1'], 'value1')
        self.assertEquals(a['key_level2'], 'value2')

    def test_call_instance(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key2', 'value2'))
        d = a()
        for k, v in zip(('key1', 'key2'), ('value1', 'value2')):
            self.assertEquals(d[k], v)

    def test_remove_attribute(self):
        a = AttributesCollection()
        a.add(Attribute('key', 'value'))
        self.assertEquals(a['key'], 'value')
        a.remove(Attribute('key', 'value'))
        self.assertRaises(KeyError, lambda: a['key'])

    def test_remove_attribute_by_key(self):
        a = AttributesCollection()
        a.add(Attribute('key', 'value'))
        self.assertEquals(a['key'], 'value')
        a.remove('key')
        self.assertRaises(KeyError, lambda: a['key'])

    def test_add_attribute(self):
        a = AttributesCollection()
        a.add(Attribute('key', 'value'))
        self.assertIn('key', [i.key for i in a._attributes])

    def test_add_data_attribute(self):
        a = AttributesCollection()
        a.add_data_attribute('key', lambda: 'value')
        self.assertEquals(a['key'], 'value')

    def test_static_items(self):
        a = AttributesCollection()
        a.add_data_attribute('key1', lambda: 'value1')
        a.add(Attribute('key2', 'value2'))
        self.assertEquals(len(a.static_items()), 1)

    def test_collection_keys(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key2', 'value2'))
        self.assertEquals(len(a.keys), 2)
        for k in a.keys:
            self.assertIn(k, ('key1', 'key2'))

    def test_collection_values(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key2', 'value2'))
        self.assertEquals(len(a.values), 2)
        for k in a.values:
            self.assertIn(k, ('value1', 'value2'))

    def test_collection_dict_property(self):
         a = AttributesCollection()
         a.add(Attribute('key2', 'value2'))
         a.add(Attribute('key1', 'value1'))
         d = a.dict
         self.assertEquals(len(d.items()), 2)
         self.assertEquals(d['key1'], 'value1')
         self.assertEquals(d['key2'], 'value2')

         # Ordered dict should be alphabetical - reverse of the order adding in
         # this case.
         self.assertEquals(list(d.keys())[0], 'key1')
