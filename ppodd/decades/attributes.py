import collections
import importlib
import warnings

STR_DERIVED_FROM_FILE = '<derived from file>'
ATTRIBUTE_NOT_SET = 'ATTRIBUTE_NOT_SET'


class AttributeNotSetError(Exception):
    """
    An exception which should be raised if a required attribute is not set.
    """


class NonStandardAttributeError(Exception):
    """
    An exception which may be raised if an attribute is added to an
    AttributesCollection which is not defined in a standard.
    """


class AttributesCollection(object):
    """
    An attributes collection is a collection of metadata key-value pairs, which
    nominally correspond to global and/or variable attributes in a netCDF file.
    Each AttributesCollection is associated with a definition and version,
    which defines which attributes are required or optional.
    """

    def __init__(self, dataset=None, definition=None, version=1.0, strict=True):
        """
        Initialise an instance.

        Kwargs:
            dataset: a ppodd Dataset associated with this attributes
                     collection.
            definition: a string pointing to the classpath of an attributes
                        definition.
            version: the version of the definition to adhere to.
            strict: if True do not allow attributes which are not defined in
                    the definition to be added to the collection.
        """
        # Init instance variables
        self._dataset = dataset
        self._attributes = []
        self._data_attributes = {}
        self._compliance = False
        self._definition = definition
        self._strict = strict

        # Get the definition from the classpath
        if isinstance(definition, str):
            _def_module, _def_var = definition.rsplit('.', 1)
            definition = getattr(importlib.import_module(_def_module), _def_var)

        if definition is not None:
            # Set REQUIRED and OPTIONAL attributes from the attributes definition
            self.REQUIRED_ATTRIBUTES = [
                g for g in definition.keys() if definition[g]['required']
                and version in definition[g]['versions']
            ]

            self.OPTIONAL_ATTRIBUTES = [
                g for g in definition.keys() if not definition[g]['required']
                and version in definition[g]['versions']
            ]
        else:
            self.REQUIRED_ATTRIBUTES = []
            self.OPTIONAL_ATTRIBUTES = []
            self._strict = False

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
                return self._compliancify(g)

        for _key, _value in self._data_attributes.items():
            if _key == key:
                return self._compliancify(Attribute(_key, _value))

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
        if self._compliance:
            if callable(att.value) or att.value == ATTRIBUTE_NOT_SET:
                return STR_DERIVED_FROM_FILE
        else:
            if callable(att.value):
                try:
                    return att.value()
                except Exception:
                    return None
            else:
                return att.value

        return att.value

    def set_compliance_mode(self, comp):
        """
        Set compliance mode. TODO: why isn't this done with properties??

        Args:
            comp: compliance mode, expected as a boolean.
        """
        self._compliance = comp

    def remove(self, att):
        """
        Remove an attribute from the AttributesCollection

        Args:
            att: the Attribute or key of Attribute to remove.
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
            att: The Attribute to add the the collection

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

        # Add the attribute
        self._attributes.append(att)

    @property
    def strict(self):
        """
        Get strict mode. If strict is truthy, attributes not defined in the
        standard cannot be added to the collection.

        Returns:
            self._strict
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
        Return True if the AttributesCollection is compliant. In this context
        this means that all REQUIRED_ATTRIBUTES are set.
        """
        for required in self.REQUIRED_ATTRIBUTES:
            if self[required] == ATTRIBUTE_NOT_SET:
                return False
        return True

    def add_data_attribute(self, param, attrs):
        """
        Add a data attribute - an attribute that is callable to return its
        value.

        Args:
            param: the key of the data attribute.
            attrs: the callable which yields the value of the attribute.
        """
        self._data_attributes[param] = attrs

    def static_items(self):
        """
        Return a dict containing all of the attributes which are fixed at
        definition time (i.e. those which do not provide their value via a
        call.
        """
        return self._as_dict(dynamic=False).items()

    def _as_dict(self, dynamic=True):
        """
        Return a dict of all of the attributes in the AttributesCollection,
        optionally excluding those which provide their value through a
        callable.

        Kwargs:
            dynamic: if True (default), include attributes which provide their
                     value through a call. If False, exclude these

        Returns:
            a dictionary of attributes, optionally excluding those providing
            their value through a call.
        """
        from ppodd.decades import DecadesDataset, DecadesVariable
        _dict = {}
        for glo in self._attributes:
            _dict[glo.key] = self._compliancify(glo)

        if not dynamic:
            return _dict

        for name, _pack in self._data_attributes.items():
            if isinstance(self._dataset, DecadesDataset):
                key, attrs = _pack
                try:
                    var = self._dataset[key]
                except KeyError:
                    if self._compliance:
                        _dict[name] = STR_DERIVED_FROM_FILE
                    continue

            if isinstance(self._dataset, DecadesVariable):
                attrs = _pack
                var = self._dataset()
                if self._compliance:
                    _dict[name] = STR_DERIVED_FROM_FILE
                    continue

            for _attr in attrs:
                var = getattr(var, _attr)

            _dict[name] = self._compliancify(Attribute(name, var))

        return _dict

    @property
    def keys(self):
        """
        Return dict_keys of all of the keys in the AttributesCollection.
        """
        return self.dict.keys()

    @property
    def values(self):
        """
        Return dict_values of all of the values in the AttributesCollection.
        """
        return self.dict.values()

    @property
    def dict(self):
        """
        Return an OrderedDict of all attribures in the AttributesCollection,
        ordered alphabetically by key.
        """
        _dict = self._as_dict()
        _sorted = collections.OrderedDict()
        for key in sorted(_dict.keys()):
            _sorted[key] = _dict[key]
        return _sorted


class Attribute(object):
    """
    An Attribute is a simple wrapper containing a key/value pair, and is
    essentially immutable once created.
    """

    def __init__(self, key, value):
        """
        Initialize a class instance.

        Args:
            key: the attribute key
            value: the attribute value.
        """
        self._key = key
        self._value = value

    def __repr__(self):
        return r'Attribute({!r}, {!r})'.format(self.key, self.value)

    @property
    def key(self):
        """
        Implement key as a property.
        """
        return self._key

    @property
    def value(self):
        """
        Implement value as a property.
        """
        return self._value
