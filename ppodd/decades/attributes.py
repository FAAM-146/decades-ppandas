import collections
import enum
import importlib
from typing import Any
import warnings

from dataclasses import dataclass

import pydantic

from vocal.schema_types import DerivedString, OptionalDerivedString

from ppodd import URL as PPODD_URL, DOI as PPODD_DOI

STR_DERIVED_FROM_FILE = '<derived from file>'
ATTRIBUTE_NOT_SET = 'ATTRIBUTE_NOT_SET'
ATTR_USE_EXAMPLE = '<example>'


class AttributeNotSetError(Exception):
    """
    An exception which should be raised if a required attribute is not set.
    """


class NonStandardAttributeError(Exception):
    """
    An exception which may be raised if an attribute is added to an
    AttributesCollection which is not defined in a standard.
    """

@dataclass
class DocAttribute:
    value: Any
    doc_value: Any

    def __getattr__(self, name: str) -> Any:
        """
        Pass any attribute requests to the value of the attribute.

        Args:
            name: the name of the attribute to get

        Returns:
            the value of the attribute
        """
        return getattr(self.value, name)


class AttributesCollection(object):
    """
    An attributes collection is a collection of metadata key-value pairs, which
    nominally correspond to global and/or variable attributes in a netCDF file.
    Each AttributesCollection is associated with a definition and version,
    which defines which attributes are required or optional.
    """

    def __init__(self, dataset=None, definition=None, strict=True):
        """
        Initialise an instance.

        Args:
            dataset (DecadesDataset): a ppodd Dataset associated with this
                attributes collection.
            definition (str, optional): a string pointing to the classpath of
                an attributes definition.
            version (float, optional): the version of the definition to adhere
                to. Defaults to 1.0.
            strict (bool, optional): if True do not allow attributes which are
                not defined in the definition to be added to the collection.
        """
        # Init instance variables
        self._dataset = dataset
        self._attributes = []
        self._data_attributes = {}
        self._compliance = False

        self._strict = strict

        _definition = None
        # Get the definition from the classpath
        if isinstance(definition, str):
            _def_module, _def_var = definition.rsplit('.', 1)
            _definition = getattr(importlib.import_module(_def_module), _def_var)
        elif definition is not None:
            _definition = definition

        try:
            # Set REQUIRED and OPTIONAL attributes from the attributes definition
            schema = _definition.schema()

            self.REQUIRED_ATTRIBUTES = [
                g for g in schema['properties'].keys() if g in schema['required']
            ]

            self.OPTIONAL_ATTRIBUTES = [
                g for g in schema['properties'].keys() if g not in schema['required']
            ]

        except AttributeError:
            _definition = {}
            self.REQUIRED_ATTRIBUTES = []
            self.OPTIONAL_ATTRIBUTES = []
            self._strict = False

        self._definition = _definition

        # Create placeholders for all of the required attributes
        for key in self.REQUIRED_ATTRIBUTES:
            self.add(Attribute(key, ATTRIBUTE_NOT_SET))

    def __getitem__(self, key):
        """
        Implement [].

        Args:
            key: the get key to search for

        Returns:
            the value of key, if key is set

        Raises:
            KeyError: if key is not found in the attributes collection
        """
        for g in self._attributes:
            if g.key == key:
                # return g
                return self._compliancify(g)

        # for _key, _value in self._data_attributes.items():
        #     if _key == key:
        #         return self._compliancify(Attribute(_key, _value))

        raise KeyError('{} not an attribute'.format(key))

    def __setitem__(self, key, value):
        """
        Implement [x].

        Args:
            key: the key(name) of the attribute to set
            value: the value to associate with key
        """
        self.remove(Attribute(key, None))

        if type(value) is dict:
            for _k, _v in value.items():
                __k = '_'.join((key, _k))
                self.add(Attribute(__k, _v))
            return

        if value == ATTR_USE_EXAMPLE:
            value = self._definition.schema()['properties'][key]['example']

        self.add(Attribute(key, value))

    def __call__(self):
        """
        Implement (). Calling a class instance returns a dictionary of all of
        the attributues.

        Retuns:
            a dict of all of the attributes in the AttributesCollection.
        """
        return self.dict

    def _compliancify(self, att):
        if callable(att.value):
            if getattr(self._dataset, 'doc_mode', False):
                try:
                    return att.doc_value
                except AttributeError:
                    pass
            try:
                return att.value()
            except Exception:
                return None
        else:
            return att.value

    def remove(self, att):
        """
        Remove an attribute from the AttributesCollection

        Args:
            att (:obj:`Attribute` or :obj:`str`): the Attribute or key of
                Attribute to remove.
        """
        if isinstance(att, str):
            # Assume only the key is given
            att = Attribute(att, None)

        for _att in self._attributes:
            if _att.key == att.key:
                self._attributes.remove(_att)
                return

    def add(self, att):
        """
        Add an Attribute to the AttributesCollection.

        Args:
            att (Attribute): The Attribute to add the the collection

        Raises:
            TypeError: if att is not of type Attribute.
            NonStandardAttributeError: if strict mode is on and att not in
                                       definition
        """

        # Type checking
        if not isinstance(att, Attribute):
            raise TypeError('attributes must be of type <Attribute>')

        # Remove duplicate keys
        self.remove(att)

        # Raise error or warning if the attribute is not present in the
        # definition
        if att.key not in self.REQUIRED_ATTRIBUTES + self.OPTIONAL_ATTRIBUTES:
            _message = f'Attribute \'{att.key}\' is not defined in standard'
            if self.strict:
                raise NonStandardAttributeError(_message)

        if att.value == ATTR_USE_EXAMPLE:
            att = Attribute(
                att.key,
                self._definition.schema()['properties'][att.key]['example']
            )

        # Add the attribute
        self._attributes.append(att)

    @property
    def strict(self):
        """
        bool: Get strict mode. If strict is truthy, attributes not defined in
        the standard cannot be added to the collection.
        """
        return self._strict

    @strict.setter
    def strict(self, strict):
        """
        Set the strict property.

        Args:
            strict: if truthy, set strict mode to on, otherwise off.
        """
        self._strict = bool(strict)

    @property
    def is_compliant(self):
        """
        bool: True if the AttributesCollection is compliant. In this context
        this means that all REQUIRED_ATTRIBUTES are set.
        """
        for required in self.REQUIRED_ATTRIBUTES:
            if self[required] == ATTRIBUTE_NOT_SET:
                return False
        return True

    def static_items(self):
        """
        Returns:
            dict: a dict containing all of the attributes which are fixed at
            definition time (i.e. those which do not provide their value via a
            call.
        """
        return self._as_dict(dynamic=False).items()

    def _as_dict(self, dynamic=True):
        """
        Return a dict of all of the attributes in the AttributesCollection,
        optionally excluding those which provide their value through a
        callable.

        Args:
            dynamic (bool, optional): if True (default), include attributes
                which provide their value through a call. If False, exclude
                these.

        Returns:
            dict: a dictionary of attributes, optionally excluding those
                providing their value through a call.
        """
        
        _dict = {}

        for glo in self._attributes:
            if not dynamic:
                if (glo._context is not None) or callable(glo.value):
                    continue

            doc_mode = getattr(self._dataset, 'doc_mode', False)

            if doc_mode:
                try:
                    _value = glo.doc_value
                except AttributeError:
                    _value = glo.value
            else:
                _value = glo.value

            while callable(_value):
                _value = _value()

            _dict[glo.key] = _value

        return _dict

    @property
    def keys(self):
        """
        dict_keys: Return dict_keys of all of the keys in the
        :obj:`AttributesCollection`.
        """
        return self.dict.keys()

    @property
    def values(self):
        """
        dict_values: Return dict_values of all of the values in the
        :obj:`AttributesCollection`.
        """
        return self.dict.values()

    @property
    def dict(self):
        """
        collections.OrderedDict: Return an OrderedDict of all attribures in the
        :obj:`AttributesCollection`, ordered alphabetically by key.
        """
        _dict = self._as_dict()
        _sorted = collections.OrderedDict()
        for key in sorted(_dict.keys()):
            _sorted[key] = _dict[key]
        return _sorted


class Context(enum.Enum):
    ITEM = enum.auto()
    ATTR = enum.auto()
    DATA = enum.auto()
   

class Attribute(object):
    """
    An Attribute is a simple wrapper containing a key/value pair, and is
    considered immutable once created.
    """

    def __init__(self, key, value, context=None, context_type=Context.ATTR):
        """
        Initialize a class instance.

        Args:
            key (str): the attribute key
            value (Object): the attribute value.
        """
        self._key = key
        if isinstance(value, DocAttribute):
            self._value = value.value
            self.doc_value = value.doc_value
        else:
            self._value = value
        self._context = context
        self._context_type = context_type

    def __repr__(self):
        return r'Attribute({!r}, {!r})'.format(self.key, self.value)

    @property
    def key(self):
        """
        str: The Attribute key.
        """
        return self._key

    @property
    def value(self):
        """
        Object: the Attribute value.
        """
        
        if self._context is None:
            return self._value

        if self._context_type == Context.ATTR:
            try:
                return getattr(self._context, self._value)
            except AttributeError:
                return None

        if self._context_type == Context.ITEM:
            try:
                return self._context[self._value]
            except KeyError:
                return None

        if self._context_type == Context.DATA:
            key, *attrs = self._value
            try:
                value = self._context[key]
            except KeyError:
                return None

            for attr in attrs:
                value = getattr(value, attr)
                if callable(value):
                    value = value()
            
            return value

GLOBALS = {
    'core': {
        'comment': OptionalDerivedString,
        'constants_file': DerivedString,
        'creator_url': 'https://www.faam.ac.uk',
        'processing_software_commit': DerivedString,
        'processing_software_version': DerivedString,
        'processing_software_doi': PPODD_DOI,
        'processing_software_url': PPODD_URL,
        'project_acronym': DerivedString,
        'project_name': DerivedString,
        'project_principal_investigator': DerivedString,
        'project_principal_investigator_email': DerivedString,
        'project_principal_investigator_url': DerivedString,
        'revision_comment': OptionalDerivedString,
        'time_coverage_start': DerivedString,
        'time_coverage_end': DerivedString,
        'time_coverage_duration': DerivedString,
        'metadata_link': 'https://github.com/FAAM-146/faam-data/'



    }
}