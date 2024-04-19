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
    Required: str = Field(
        description='A required attribute', 
        json_schema_extra={
            'example': 'my attribute'
        })
    RequiredDefault: str = Field(
        description='A required attribute w/ default',
        json_schema_extra={
            'ppodd_default': 'A required attribute'
        }
    )
    OptionalAttr: Optional[str] = 'An optional attribute'

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
        self.assertEqual(a.REQUIRED_ATTRIBUTES, [])

    def test_optional_attributes_with_no_definition(self):
        a = AttributesCollection()
        self.assertEqual(a.OPTIONAL_ATTRIBUTES, [])

    def test_get_item_non_compliancify(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key2', 'value2'))
        self.assertEqual(a['key1'], 'value1')
        self.assertEqual(a['key2'], 'value2')

    def test_add_duplicate_key(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key1', 'value2'))
        self.assertEqual(a['key1'], 'value2')

    def test_add_invalid_key_in_strict_mode(self):
        a = AttributesCollection(definition=Attributes)
        self.assertRaises(
            NonStandardAttributeError,
            lambda: a.add(Attribute('invalid', 'invalid'))
        )

    def test_set_item(self):
        a = AttributesCollection()
        a['key1'] = 'value'
        self.assertEqual(a['key1'], 'value')

    def test_set_item_as_dict(self):
        a = AttributesCollection()
        a['key'] = {'level1': 'value1', 'level2': 'value2'}
        self.assertEqual(a['key_level1'], 'value1')
        self.assertEqual(a['key_level2'], 'value2')

    def test_call_instance(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key2', 'value2'))
        d = a()
        for k, v in zip(('key1', 'key2'), ('value1', 'value2')):
            self.assertEqual(d[k], v)

    def test_remove_attribute(self):
        a = AttributesCollection()
        a.add(Attribute('key', 'value'))
        self.assertEqual(a['key'], 'value')
        a.remove(Attribute('key', 'value'))
        self.assertRaises(KeyError, lambda: a['key'])

    def test_remove_attribute_by_key(self):
        a = AttributesCollection()
        a.add(Attribute('key', 'value'))
        self.assertEqual(a['key'], 'value')
        a.remove('key')
        self.assertRaises(KeyError, lambda: a['key'])

    def test_add_attribute(self):
        a = AttributesCollection()
        a.add(Attribute('key', 'value'))
        self.assertIn('key', [i.key for i in a._attributes])

    def test_add_data_attribute(self):
        a = AttributesCollection()
        a.add(Attribute('key', lambda: 'value'))
        self.assertEqual(a['key'], 'value')

    def test_static_items(self):
        a = AttributesCollection()
        a.add(Attribute('key1', lambda: 'value1'))
        a.add(Attribute('key2', 'value2'))
        self.assertEqual(len(a.static_items()), 1)

    def test_collection_keys(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key2', 'value2'))
        self.assertEqual(len(a.keys), 2)
        for k in a.keys:
            self.assertIn(k, ('key1', 'key2'))

    def test_collection_values(self):
        a = AttributesCollection()
        a.add(Attribute('key1', 'value1'))
        a.add(Attribute('key2', 'value2'))
        self.assertEqual(len(a.values), 2)
        for k in a.values:
            self.assertIn(k, ('value1', 'value2'))

    def test_collection_dict_property(self):
         a = AttributesCollection()
         a.add(Attribute('key2', 'value2'))
         a.add(Attribute('key1', 'value1'))
         d = a.dict
         self.assertEqual(len(d.items()), 2)
         self.assertEqual(d['key1'], 'value1')
         self.assertEqual(d['key2'], 'value2')

         # Ordered dict should be alphabetical - reverse of the order adding in
         # this case.
         self.assertEqual(list(d.keys())[0], 'key1')
