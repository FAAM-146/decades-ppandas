import collections
import warnings

STR_DERIVED_FROM_FILE = '<derived from file>'
ATTRIBUTE_NOT_SET = 'ATTRIBUTE_NOT_SET'


class AttributeNotSetError(Exception):
    pass


class NonStandardAttributeError(Exception):
    pass


class AttributesCollection(object):

    def __init__(self, dataset=None, definition=None, version=1.0, strict_mode=True):

        self.REQUIRED_ATTRIBUTES = [
            g for g in definition.keys() if definition[g]['required']
            and version in definition[g]['versions']
        ]

        self.OPTIONAL_ATTRIBUTES = [
            g for g in definition.keys() if not definition[g]['required']
            and version in definition[g]['versions']
        ]

        self._dataset = dataset
        self._attributes = []
        self._data_attributes = {}
        self._compliance = False
        self._definition = definition
        self._strict_mode = strict_mode

        for key in self.REQUIRED_ATTRIBUTES:
            self.add(Attribute(key, ATTRIBUTE_NOT_SET))

    def __getitem__(self, key):
        for g in self._attributes:
            if g.key == key:
                return self._compliancify(g)

        raise KeyError('{} not an attribute'.format(key))

    def __setitem__(self, key, value):
        if type(value) is dict:
            for _k, _v in value.items():
                __k = '_'.join((key, _k))
                self[__k] = _v

        else:
            self.add(Attribute(key, value))

    def __call__(self):
        return self.dict

    def _compliancify(self, att):
        if self._compliance:
            if callable(att.value) or att.value == ATTRIBUTE_NOT_SET:
                return STR_DERIVED_FROM_FILE
        else:
            if callable(att.value):
                return att.value()
            else:
                return att.value

        return att.value

    def set_compliance_mode(self, comp):
        self._compliance = comp

    def add(self, att):
        if not isinstance(att, Attribute):
            raise TypeError('attributes must be of type <Attribute>')

        for i in self._attributes:
            if i.key == att.key:
                self._attributes.remove(i)
                break

        if att.key not in self.REQUIRED_ATTRIBUTES + self.OPTIONAL_ATTRIBUTES:
            _message = f'Attribute \'{att.key}\' is not defined in standard'
            if self.strict_mode:
                raise NonStandardAttributeError(_message)
            else:
                warnings.warn(_message)

        self._attributes.append(att)

    @property
    def strict_mode(self):
        return self._strict_mode

    @strict_mode.setter
    def strict_mode(self, strict):
        self._strict_mode = bool(strict)

    @property
    def is_compliant(self):
        for required in self.REQUIRED_ATTRIBUTES:
            if self[required] == ATTRIBUTE_NOT_SET:
                return False
        return True

    def add_data_attribute(self, param, attrs):
        self._data_attributes[param] = attrs

    def static_items(self):
        return self._as_dict(dynamic=False).items()

    def _as_dict(self, dynamic=True):
        _dict = {}
        for glo in self._attributes:
            _dict[glo.key] = self._compliancify(glo)

        if not dynamic:
            return _dict

        for name, _pack in self._data_attributes.items():
            key, attrs = _pack
            try:
                var = self._dataset[key]
            except KeyError:
                if self._compliance:
                    _dict[name] = STR_DERIVED_FROM_FILE
                continue

            for _attr in attrs:
                var = getattr(var, _attr)

            _dict[name] = self._compliancify(Attribute(name, var))

        return _dict

    @property
    def keys(self):
        return self.dict.keys()

    @property
    def values(self):
        return self.dict.values()

    @property
    def dict(self):
        _dict = self._as_dict()
        _sorted = collections.OrderedDict()
        for key in sorted(_dict.keys()):
            _sorted[key] = _dict[key]
        return _sorted


class Attribute(object):
    def __init__(self, key, value):
        self._key = key
        self._value = value

    def __repr__(self):
        return r'Attribute({!r}, {!r})'.format(self.key, self.value)

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value
